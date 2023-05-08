"""Training entrance for Chinese LLama.

Use Trainer for model traning.
Use Generator for testing language model performance

"""

import os
import torch
import torch.distributed as dist
import json
from pathlib import Path


from torch.utils.tensorboard import SummaryWriter

from model import ModelArgs, LLaMATransformer
from data import build_train_valid_test_datasets
from utils.utils import print_rank_0, clip_grad_norm
from utils.initialize import (initialize_model_parallel, get_model_parallel_world_size, 
                            get_model_parallel_rank, get_model_parallel_group,
                            get_data_parallel_rank, get_data_parallel_group,
                            get_model_parallel_src_rank, get_data_parallel_world_size)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self._setup_model_parallel()
        if not os.path.exists(self.args.logdir): 
            os.makedirs(self.args.logdir, exist_ok=True)
        self.summary_writer = SummaryWriter(self.args.logdir)
    
    def _setup_model_parallel(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        if torch.cuda.is_available():
            torch.distributed.init_process_group("nccl")
            torch.cuda.set_device(local_rank)
        else:
            torch.distributed.init_process_group("gloo")

        # 下列函数决定了model_parallel, pipeline parallel, data parallel size (自动确定的)
        initialize_model_parallel(model_parallel_size_=self.args.model_parallel_size, pipeline_length=1)

        # seed must be the same in all processes
        torch.manual_seed(self.args.seed)
        return local_rank, world_size
    
    def _get_data(self):
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.data import DataLoader

        data_group_rank = get_data_parallel_rank()
        start_step = self.args.training_config.start_step # iterate data from the start_step (for resuming training)
        print_rank_0('> building train, validation, and test datasets from step {}'.format(start_step))
        train_ds, eval_ds, test_ds = build_train_valid_test_datasets(self.args.data_path, 
                                            self.args.splits_string, self.args.model_config.max_seq_len)
        if start_step > 0:
            start_idx = start_step*self.args.training_config.batch_size
            train_ds.tokens = train_ds.tokens[start_idx:]
        train_data_sampler = DistributedSampler(train_ds, num_replicas=get_data_parallel_world_size(), 
                                                rank=data_group_rank, shuffle=True, seed=self.args.seed)
        train_loader = DataLoader(train_ds, batch_size=self.args.training_config.micro_batch_size, sampler=train_data_sampler)

        eval_data_sampler = DistributedSampler(eval_ds, num_replicas=get_data_parallel_world_size(), 
                                                rank=data_group_rank, shuffle=False, seed=self.args.seed)
        eval_loader = DataLoader(eval_ds, batch_size=self.args.training_config.micro_batch_size, sampler=eval_data_sampler)


        test_data_sampler = DistributedSampler(test_ds, num_replicas=get_data_parallel_world_size(), 
                                                rank=data_group_rank, shuffle=False, seed=self.args.seed)
        test_loader = DataLoader(test_ds, batch_size=self.args.training_config.micro_batch_size, sampler=test_data_sampler)

        return train_loader, eval_loader, test_loader

    def _get_model(self, model_args, iter_n_step=10000):
        model = LLaMATransformer(model_args)
        if torch.cuda.is_available():
            model = model.cuda(int(os.environ['LOCAL_RANK']))
        torch.set_default_tensor_type(torch.FloatTensor)
        optimizer = torch.optim.AdamW(model.parameters(), betas=(self.args.optimizer_config.beta1, self.args.optimizer_config.beta2), eps=1e-08, 
                            weight_decay=float(self.args.optimizer_config.weight_decay), lr=float(self.args.optimizer_config.lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_n_step, eta_min=float(self.args.optimizer_config.min_lr))

        return model, optimizer, scheduler

    def _save_checkpoint(self, model, optimizer, scheduler, n_step):
        if not os.path.exists(self.args.ckpt_dir):
            os.makedirs(self.args.ckpt_dir, exist_ok=True)
        local_data_parallal_rank = get_data_parallel_rank()
        if local_data_parallal_rank == 0:
            local_rank = get_model_parallel_rank()
            checkpoint_path = os.path.join(self.args.ckpt_dir, "model_rank_{}_step_{}.pth".format(local_rank, n_step))
            model_params_path = os.path.join(self.args.ckpt_dir, "params.json")
            print("Saving model on rank {} with step {}...".format(local_rank, n_step))
            if torch.distributed.get_rank() == 0: # save hyper-parameters with main process
                with open(model_params_path, "w+") as fout:
                    fout.write(json.dumps(model.params.__dict__))
            torch.set_default_tensor_type(torch.FloatTensor)
            checkpoint = { 
                'step': n_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler
            }
            torch.save(checkpoint, checkpoint_path)

    def _delete_checkpoint(self, n_step):
        # delete the out dated checkpoints
        local_data_parallal_rank = get_data_parallel_rank()
        if local_data_parallal_rank == 0:
            local_rank = get_model_parallel_rank()
            checkpoint_path = os.path.join(self.args.ckpt_dir, "model_rank_{}_step_{}.pth".format(local_rank, n_step))
            try:
                os.remove(checkpoint_path)
            except:
                "Fail to remove checkpointed model at {}!".format(checkpoint_path)

    def _load_checkpoint(self):
        # restart training from the start_step (for resuming training)
        local_rank = get_model_parallel_rank()
        print("Loading")
        # load model config
        with open(Path(self.args.ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(**params)
        self.args.model_config = model_args
        model, optimizer, scheduler = self._get_model(model_args, iter_n_step=10000)
        # torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # torch.set_default_tensor_type(torch.FloatTensor)
        local_rank = get_model_parallel_rank()
        checkpoint_path = os.path.join(self.args.ckpt_dir, "model_rank_{}_step_{}.pth".format(local_rank, 
                                                                        self.args.training_config.resume_step))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        old_scheduler = checkpoint["lr_scheduler"]
        scheduler.load_state_dict(old_scheduler.state_dict())
        return model, optimizer, scheduler

    def train(self):
        train_loader, eval_loader, _ = self._get_data()
        # do not use len(train_loader) to get the batch num due to the introduction of micro_batch_size
        iter_n_step = int(len(train_loader.dataset) / self.args.training_config.batch_size) * self.args.training_config.n_epochs
        print_rank_0("Expected iteration number: {}".format(iter_n_step))
        # 用config 参数覆盖默认参数
        if self.args.training_config.resume_step > 0:
            model, optimizer, scheduler = self._load_checkpoint()
        else:
            model_args: ModelArgs = ModelArgs(**self.args.model_config) 
            model, optimizer, scheduler = self._get_model(model_args, iter_n_step)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda last_epoch: 1. * last_epoch / self.args.training_config.warmup_step)
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = dist.get_rank()
        def average_gradients(model):
            if get_data_parallel_world_size() > 1:
                for param in model.parameters():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=get_data_parallel_group())
                    param.grad.data /= get_data_parallel_world_size()
            
            # Clipping gradients helps prevent the exploding gradient.
            if self.args.optimizer_config.clip_grad > 0:
                return clip_grad_norm(model.parameters(), self.args.optimizer_config.clip_grad)

        model.train()
        import time
        # ---------------------- Perform training ----------------------- #
        cum_loss = 0.0
        # gradient accumulate step
        gas = self.args.training_config.batch_size // (self.args.training_config.micro_batch_size * get_data_parallel_world_size())
        step = self.args.training_config.resume_step
        micro_step = 0
        optimizer.zero_grad()
        keep_checkpoint_steps = []
        print_rank_0('gas step:{}'.format(gas))
        start_time = time.time()
        for epoch in range(self.args.training_config.n_epochs):
            train_loader.sampler.set_epoch(epoch=epoch)
            for batched_data in train_loader:
                tokens = batched_data['text']
                if torch.cuda.is_available():
                    tokens = tokens.cuda(local_rank)
                micro_step += 1
                # with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(tokens, tokens)
                # reduce loss across multiple data_parall group
                loss = outputs['loss']
                # model parallel group中的所有进程进行反向传播，这会放大gradient model_parall_size倍
                (loss / get_model_parallel_world_size() / gas).backward()
                # 归并数据并行中的其他进程
                """Reduce a tensor of losses across all GPUs."""
                reduced_loss = loss.clone().detach().view(1)
                if get_data_parallel_world_size() > 1:
                    dist.all_reduce(reduced_loss, group=get_data_parallel_group())
                reduced_loss = reduced_loss / get_data_parallel_world_size()
                cum_loss += reduced_loss.item()
                if micro_step % gas == 0:
                    # 归并其他进程中的梯度以更新模型参数 
                    grad_norm = average_gradients(model) 
                    optimizer.step()
                    optimizer.zero_grad() 
                    step += 1
                    cum_loss /= gas
                    
                    if global_rank == 0 and step % self.args.training_config.log_every_n_step == 0:
                        print_rank_0('train loss at step {}: {}'.format(step, cum_loss))
                        self.summary_writer.add_scalar('train/grad_norm', grad_norm, step)
                        self.summary_writer.add_scalar('train/loss', cum_loss, step)
                        self.summary_writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], step)
                        end_time = time.time()
                        print_rank_0('It costs {}s to step {} for {} samples!'.format(end_time - start_time, step, step * self.args.training_config.batch_size))
                    if step >= self.args.training_config.start_eval_step and step % self.args.training_config.eval_every_n_step == 0:
                        loss = self.eval(eval_loader, model)
                        if global_rank == 0:
                            self.summary_writer.add_scalar('eval_loss', loss, step)
                    if step % self.args.training_config.save_every_n_step == 0:
                        self._save_checkpoint(model, optimizer, scheduler, step)
                        keep_checkpoint_steps.append(step)
                        if len(keep_checkpoint_steps) > self.args.training_config.keep_n_checkpoints:
                            remove_step = keep_checkpoint_steps.pop(0)
                            self._delete_checkpoint(remove_step)
                    if step < self.args.training_config.warmup_step:
                        warmup_scheduler.step()
                    else:
                        scheduler.step()
                    cum_loss = 0.
    
    def eval(self, eval_loader, model):
        model.eval()
        # ---------------------- Perform training ----------------------- #
        cum_loss = 0.0
        micro_step = 0
        local_rank = int(os.environ['LOCAL_RANK'])
        print_rank_0('Evaluating ...')
        with torch.no_grad(): 
            for data in eval_loader:
                tokens = data['text']
                if torch.cuda.is_available():
                    tokens = tokens.cuda(local_rank)
                micro_step += 1
                outputs = model(tokens, tokens)
                # reduce loss across multiple data_parall group
                loss = outputs['loss']
                # 归并数据并行中的其他进程
                """Reduce a tensor of losses across all data group processes."""
                reduced_loss = loss.detach().view(1)
                if get_data_parallel_world_size() > 1:
                    torch.distributed.all_reduce(reduced_loss, group=get_data_parallel_group())
                reduced_loss = reduced_loss / get_data_parallel_world_size()
                cum_loss += reduced_loss.item()
        model.train()
        print_rank_0('Evaluating Done!')
        return cum_loss / micro_step
    
    
class Generator(Trainer):
    """
    Build a server/terminal service for generation
    """
    def __init__(self, args, max_gen_len=512, temperature=0., top_p=0.9, **kwargs):
        super().__init__(args)
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        import sentencepiece as spm
        self.model, _, _ = self._load_checkpoint()
        self.model_params = self.model.params
        
        from tokenizer import JiebaTokenizer
        self.tokenizer = JiebaTokenizer(self.args.tokenizer_path)
        # self.tokenizer = spm.SentencePieceProcessor()
        # self.tokenizer.Load(self.args.tokenizer_path)

    def generate(
        self,
        prompts,
        max_gen_len=None,
        temperature=None,
        top_p=None):

        def sample_top_p(probs, p):
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
            return next_token

        max_gen_len = max_gen_len if max_gen_len is not None else self.max_gen_len
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        bsz = len(prompts)
        prompt_tokens = [[self.tokenizer.bos_id()]+self.tokenizer.encode(x) for x in prompts]
        print_rank_0(self.tokenizer.decode(prompt_tokens[0]))

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(self.model_params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), 0).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != 0
        start_pos = min_prompt_size
        for cur_pos in range(start_pos, total_len):
            logits = self.model(tokens[:, :cur_pos])['logits']
            logits = logits[:, -1]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id())]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


if __name__ == "__main__":
    import yaml
    from munch import munchify
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./config.yaml"), "r") as fin:
        args = yaml.safe_load(fin)
        args = munchify(args)
    assert args.training_config.batch_size % args.training_config.micro_batch_size == 0, "Micro batch size should be divided by batch size!"
    trainer = Trainer(args)
    trainer.train()
    generator = Generator(args)
    while True:
        try:
            prompts = input('Input the prefix:\n')
        except:
            continue
        out = generator.generate([prompts], max_gen_len=64)
        print_rank_0(out)
