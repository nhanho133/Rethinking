# class CLIP_Clean_Train():
#     # def __init__(self, rank,local_rank,args):
#     def __init__(self, args):
#         # self.rank=rank
#         # self.local_rank = local_rank
#         self.base_model = args.base_model
#         self.model, _ = longclip.load_from_clip(self.base_model, device='cpu',download_root=args.download_root)
#         self.model.train()
#         self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
#         self.model = self.model.cuda()
        
#         self.batch_size = args.batch_size
#         self.num_epoch = args.epochs
#         self.lr = args.lr
#         self.weight_decay = args.weight_decay
#         self.warmup_length = args.warmup_length
#         if args.exp_name == "auto":
#             self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
#         else:
#             self.logdir = args.exp_name
#         self.ckptdir = self.logdir + "/ckpt/"
#         os.makedirs(self.ckptdir, exist_ok=True)
#         self.writer = SummaryWriter(self.logdir)

#         # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
           
#         self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         self.scaler =GradScaler()

#     def split_into_detail_captions(self, text_long):
#         """
#         Nhận vào danh sách long caption (mỗi phần tử là chuỗi) và tách thành danh sách các detail caption dựa trên dấu chấm.
#         Ví dụ:
#           Input: ["A cat sitting on a mat. The cat is playing with a ball. Clear sky."]
#           Output: [["A cat sitting on a mat", "The cat is playing with a ball", "Clear sky"]]
#         """
#         detail_caps = [p.strip() for p in text_long.split('.') if p.strip()]
#         return detail_caps
    
#     def extract_image_patches(self, images, patch_size=16):
#         """
#         Tách ảnh thành các patch.
#         Args:
#             images: Tensor có shape [B, C, H, W]
#             patch_size: Kích thước patch (giả sử stride = patch_size)
#         Returns:
#             patches: Tensor có shape [B, num_patches, C, patch_size, patch_size]
#         """
#         B, C, H, W = images.shape
#         # Sử dụng unfold để tách ảnh thành các patch theo chiều H và W
#         patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         # patches shape lúc này: [B, C, num_patches_H, num_patches_W, patch_size, patch_size]
#         patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
#         # Đưa về dạng [B, num_patches, C, patch_size, patch_size]
#         patches = patches.permute(0, 2, 1, 3, 4)
#         return patches


#     def train_epoch(self, dataloader, epoch, start_iter=0):
#         running_loss = 0.0
#         running_loss_short = 0.0
#         #rank = torch.distributed.get_rank() 
#         num_batches_per_epoch = len(dataloader)
#         # for i, (images, texts, short_text) in enumerate(tqdm(dataloader, disable=(self.rank != 0))):
#         for i, (images, texts, short_text) in enumerate(tqdm(dataloader)):
#             step = num_batches_per_epoch * epoch + i
#             if step < start_iter:
#                 continue
            
#             detail_caps = [self.split_into_detail_captions(text) for text in texts]
#             # import pdb
#             # pdb.set_trace()
#             tokenized_detail_caps = [
#                 [longclip.tokenize(dc, truncate=True).cuda() for dc in self.split_into_detail_captions(text)]
#                 for text in texts
#             ]

#             texts = longclip.tokenize(texts, truncate=True).cuda()
#             short_text = longclip.tokenize(short_text, truncate=True).cuda()
#             self.scheduler(step)
#             self.optimizer.zero_grad()

#             # Tách ảnh thành patch
#             image_patches = self.extract_image_patches(images, patch_size=16).cuda()


#             with torch.cuda.amp.autocast():
#                 # loss_long,loss_short = self.model(images, texts,short_text,self.rank)
#                 # loss_long,loss_short,finegrained_loss,consistency_loss = self.model(images, texts,short_text, tokenized_detail_caps, image_patches)
#                 loss_long,loss_short,finegrained_loss = self.model(images, texts,short_text, tokenized_detail_caps, image_patches)
            
#                 # loss=loss_long+loss_short+finegrained_loss+consistency_loss*0.1
#                 loss=loss_long+loss_short+finegrained_loss
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#             # print(f"Epoch [{epoch}], Step [{step}], Loss: {loss.item():.4f}, "
#             #       f"ITCL: {loss_long.item():.4f}, ITCS: {loss_short.item():.4f}, "
#             #       f"finegrained_loss: {finegrained_loss.item():.4f}, "
#             #       f"consistency_loss: {consistency_loss.item():.4f}")
#             print(f"Epoch [{epoch}], Step [{step}], Loss: {loss.item():.4f}, "
#                   f"ITCL: {loss_long.item():.4f}, ITCS: {loss_short.item():.4f}, "
#                   f"finegrained_loss: {finegrained_loss.item():.4f}")
        
       
#     @torch.no_grad()
#     def test_epoch(self, dataloader):
#         temp_corr_dict = dict()
#         # rank = torch.distributed.get_rank()

#         # for id, (images, text) in enumerate(tqdm(dataloader, disable=(rank != 0))):
#         for id, (images, text) in enumerate(tqdm(dataloader)):
#             images = images.cuda()
#             image_features = self.model.module.encode_image(images)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#             text = longclip.tokenize(text, truncate=True).cuda()
#             text_feature = self.model.module.encode_text(text)
#             text_feature /= text_feature.norm(dim=-1, keepdim=True)

#             i = 0
#             correct = 0
#             total = 0

#             for i in range(text_feature.shape[0]):
#                 text = text_feature[i]
#                 sim = text @ image_features.T
#                 sim = sim.squeeze()
#                 correct_i = torch.argmax(sim)

#                 if i==correct_i:
#                     correct = correct + 1
#                 total = total + 1

#         return correct/total
    
#     def test(self, epoch=0):
#         # rank = torch.distributed.get_rank()
#         rank = 0
#         if rank == 0:
#             self.model.eval()
#             testset = share4v_val_dataset()
#             testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=32, pin_memory=True)
#             with torch.no_grad():    

#                 acc = self.test_epoch(testloader)
#                 print("=====================================")
#                 print(f"test mean of share4v retrieval: {acc}")
#                 print("=====================================")

#             return
    
#     def train(self, resume=False, warmup_length=200):
#         trainset = share4v_train_dataset()
#         # train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
#         # train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=32, pin_memory=True)
#         train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=32, pin_memory=True)

#         self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))
#         start_epoch = 0
#         resume_iter = 0
#         # checkpoint_path = "./checkpoints/04-04--17_20_01_longclip.pt"
#         # checkpoint = torch.load(checkpoint_path, map_location='cpu')
#         #self.model.load_state_dict(checkpoint['model_state_dict'])
#         # self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

#         #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         #start_epoch = checkpoint.get('epoch', 0) + 1
#         # print(f"Resumed training from epoch")
#         for epoch in range(start_epoch, self.num_epoch):
            
#             self.train_epoch(train_loader, epoch, start_iter=resume_iter)
#             # if self.rank == 0:
#             if True:
#                 name = "longclip.pt"
#                 now = datetime.now()
#                 formatted_date = now.strftime("%m-%d--%H_%M_%S_")
#                 #torch.distributed.barrier()
#                 checkpoint_dir = "./checkpoints"
#                 os.makedirs(checkpoint_dir, exist_ok=True)
#                 torch.save(self.model.state_dict(), './checkpoints/3-'+formatted_date+name)
#                 # torch.save(self.model.module.state_dict(), './checkpoints/'+str(self.rank)+formatted_date+name)

# import os
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.cuda.amp import GradScaler
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# from datetime import datetime
# import subprocess  # For running the external eval script
# import pandas as pd  # For saving results to Excel

# class CLIP_Clean_Train():
#     def __init__(self, args):
#         self.base_model = args.base_model
#         self.model, _ = longclip.load_from_clip(self.base_model, device='cpu', download_root=args.download_root)
#         self.model.train()
#         self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
#         self.model = self.model.cuda()
        
#         self.batch_size = args.batch_size
#         self.num_epoch = args.epochs
#         self.lr = args.lr
#         self.weight_decay = args.weight_decay
#         self.warmup_length = args.warmup_length
        
#         if args.exp_name == "auto":
#             self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
#         else:
#             self.logdir = args.exp_name
#         self.ckptdir = os.path.join(self.logdir, "ckpt")
#         os.makedirs(self.ckptdir, exist_ok=True)
#         self.writer = SummaryWriter(self.logdir)

#         self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         self.scaler = GradScaler()

#         # Danh sách lưu lại kết quả của mỗi epoch (loss và eval)
#         self.results = []
        
#     def split_into_detail_captions(self, text_long):
#         """
#         Nhận vào danh sách long caption (mỗi phần tử là chuỗi) và tách thành danh sách các detail caption dựa trên dấu chấm.
#         Ví dụ:
#           Input: "A cat sitting on a mat. The cat is playing with a ball. Clear sky."
#           Output: ["A cat sitting on a mat", "The cat is playing with a ball", "Clear sky"]
#         """
#         detail_caps = [p.strip() for p in text_long.split('.') if p.strip()]
#         return detail_caps
    
#     def extract_image_patches(self, images, patch_size=16):
#         """
#         Tách ảnh thành các patch.
#         Args:
#             images: Tensor có shape [B, C, H, W]
#             patch_size: Kích thước patch
#         Returns:
#             patches: Tensor có shape [B, num_patches, C, patch_size, patch_size]
#         """
#         B, C, H, W = images.shape
#         patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
#         patches = patches.permute(0, 2, 1, 3, 4)
#         return patches

#     def train_epoch(self, dataloader, epoch, start_iter=0):
#         running_loss = 0.0
#         running_loss_long = 0.0
#         running_loss_short = 0.0
#         running_loss_finegrained = 0.0
#         num_batches_per_epoch = len(dataloader)
        
#         for i, (images, texts, short_text) in enumerate(tqdm(dataloader)):
#             step = num_batches_per_epoch * epoch + i
#             if step < start_iter:
#                 continue
            
#             # Tách detail caption cho mỗi ảnh
#             # detail_caps = [self.split_into_detail_captions(text) for text in texts]
#             # Tokenize từng detail caption của mỗi ảnh (chú ý: mỗi ảnh có thể có nhiều detail caption)
#             tokenized_detail_caps = [
#                 [longclip.tokenize(dc, truncate=True).cuda() for dc in self.split_into_detail_captions(text)]
#                 for text in texts
#             ]
#             import pdb
#             pdb.set_trace()
#             texts = longclip.tokenize(texts, truncate=True).cuda()
#             short_text = longclip.tokenize(short_text, truncate=True).cuda()
            
#             self.scheduler(step)
#             self.optimizer.zero_grad()

#             # Tách ảnh thành các patch
#             image_patches = self.extract_image_patches(images, patch_size=16).cuda()
#             import pdb
#             pdb.set_trace()
#             with torch.cuda.amp.autocast():
#                 # Gọi model trả về các loss: loss_long, loss_short và finegrained_loss
#                 import pdb
#                 pdb.set_trace()
#                 loss_long, loss_short, finegrained_loss = self.model(images, texts, short_text, tokenized_detail_caps, image_patches)
#                 # Tổng loss là tổng của 3 loss trên
#                 loss = loss_long + loss_short + finegrained_loss
                
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
            
#             print(f"Epoch [{epoch}], Step [{step}], Total Loss: {loss.item():.4f}, "
#                   f"ITCL: {loss_long.item():.4f}, ITCS: {loss_short.item():.4f}, "
#                   f"finegrained_loss: {finegrained_loss.item():.4f}")
            
#             running_loss += loss.item()
#             running_loss_long += loss_long.item()
#             running_loss_short += loss_short.item()
#             running_loss_finegrained += finegrained_loss.item()
        
#         # Tính trung bình loss qua các batch
#         avg_loss = running_loss / num_batches_per_epoch
#         avg_loss_long = running_loss_long / num_batches_per_epoch
#         avg_loss_short = running_loss_short / num_batches_per_epoch
#         avg_loss_finegrained = running_loss_finegrained / num_batches_per_epoch
        
#         print(f"========== Epoch {epoch} completed ==========")
#         print(f"Average Total Loss: {avg_loss:.4f}, Average ITCL: {avg_loss_long:.4f}, "
#               f"Average ITCS: {avg_loss_short:.4f}, Average Finegrained Loss: {avg_loss_finegrained:.4f}")
#         return avg_loss, avg_loss_long, avg_loss_short, avg_loss_finegrained

#     @torch.no_grad()
#     def test_epoch(self, dataloader):
#         # Hàm test_epoch không thay đổi
#         for id, (images, text) in enumerate(tqdm(dataloader)):
#             images = images.cuda()
#             image_features = self.model.module.encode_image(images)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#             text = longclip.tokenize(text, truncate=True).cuda()
#             text_feature = self.model.module.encode_text(text)
#             text_feature /= text_feature.norm(dim=-1, keepdim=True)

#             correct = 0
#             total = 0
#             for i in range(text_feature.shape[0]):
#                 sim = text_feature[i] @ image_features.T
#                 sim = sim.squeeze()
#                 correct_i = torch.argmax(sim)
#                 if i == correct_i:
#                     correct += 1
#                 total += 1

#         return correct / total
    
#     def test(self, epoch=0):
#         rank = 0
#         if rank == 0:
#             self.model.eval()
#             testset = share4v_val_dataset()
#             testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=32, pin_memory=True)
#             with torch.no_grad():
#                 acc = self.test_epoch(testloader)
#                 print("=====================================")
#                 print(f"Test mean of share4v retrieval: {acc}")
#                 print("=====================================")
#             return
    
#     def train(self, resume=False, warmup_length=200):
#         trainset = share4v_train_dataset()
#         train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=32, pin_memory=True)
#         self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length,
#                                    steps=self.num_epoch * len(train_loader))
#         start_epoch = 0
#         resume_iter = 0
        
#         import pdb
#         pdb.set_trace()
#         for epoch in range(start_epoch, self.num_epoch):
#             # Huấn luyện 1 epoch và lấy các loss trung bình
#             avg_loss, avg_loss_long, avg_loss_short, avg_loss_finegrained = self.train_epoch(train_loader, epoch, start_iter=resume_iter)
            
#             # Lưu checkpoint của mô hình sau mỗi epoch
#             now = datetime.now()
#             formatted_date = now.strftime("%m-%d--%H_%M_%S_")
#             checkpoint_name = f"Propose-longclip.pt"
#             torch.save(self.model.state_dict(), os.path.join(self.ckptdir, checkpoint_name))
            
#             # Chạy script eval và lấy kết quả
#             try:
#                 eval_output = subprocess.check_output(
#                     ["python", "imagenet.py"],
#                     cwd="../eval/classification/imagenet",
#                     stderr=subprocess.STDOUT
#                 )
#                 eval_result = eval_output.decode("utf-8")
#                 print("========== Evaluation Result ==========")
#                 print(eval_result)
#                 print("========================================")
#             except subprocess.CalledProcessError as e:
#                 eval_result = "Error"
#                 print("Error running the evaluation script:")
#                 print(e.output.decode("utf-8"))
            
#             # Lưu kết quả của epoch vào danh sách self.results
#             self.results.append({
#                 "epoch": epoch,
#                 "avg_total_loss": avg_loss,
#                 "avg_loss_long": avg_loss_long,
#                 "avg_loss_short": avg_loss_short,
#                 "avg_finegrained_loss": avg_loss_finegrained,
#                 "eval_result": eval_result.strip()  # loại bỏ khoảng trắng thừa và xuống dòng
#             })
            
#             # Chuyển self.results thành DataFrame và ghi vào file Excel
#             results_df = pd.DataFrame(self.results)
#             excel_file_path = os.path.join(self.logdir, "epoch_results.xlsx")
#             results_df.to_excel(excel_file_path, index=False)

# import os
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.cuda.amp import GradScaler
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader, DistributedSampler
# from tqdm import tqdm
# from datetime import datetime
# import subprocess  # For running the external eval script
# import pandas as pd  # For saving results to Excel

# # Giả sử args có các tham số: base_model, download_root, log_scale, batch_size, epochs, lr, weight_decay,
# # warmup_length, exp_name, local_rank, (v.v.). Các tham số RANK và WORLD_SIZE có thể được thiết lập qua môi trường.
# class CLIP_Clean_Train():
#     def __init__(self, args):
#         # Lấy local_rank từ args hoặc từ biến môi trường
#         self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
#         self.device = torch.device("cuda", self.local_rank)
#         torch.cuda.set_device(self.device)
        
#         # Khởi tạo process group nếu chưa được khởi tạo
#         if not dist.is_initialized():
#             dist.init_process_group(backend="nccl", init_method="env://")
        
#         self.base_model = args.base_model
#         self.model, _ = longclip.load_from_clip(self.base_model, device='cpu', download_root=args.download_root)
#         self.model.train()
#         self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
        
#         # Chuyển model sang GPU hiện tại
#         self.model = self.model.to(self.device)
#         # Bao bọc model vào DistributedDataParallel
#         self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        
#         self.batch_size = args.batch_size
#         self.num_epoch = args.epochs
#         self.lr = args.lr
#         self.weight_decay = args.weight_decay
#         self.warmup_length = args.warmup_length
        
#         # Tạo thư mục log
#         if args.exp_name == "auto":
#             self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
#         else:
#             self.logdir = args.exp_name
#         self.ckptdir = os.path.join(self.logdir, "ckpt")
#         os.makedirs(self.ckptdir, exist_ok=True)
#         # Chỉ ghi log từ process chính (local_rank == 0)
#         self.writer = SummaryWriter(self.logdir) if self.local_rank == 0 else None

#         self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         self.scaler = GradScaler()

#         # Danh sách lưu lại kết quả của mỗi epoch (loss và eval)
#         self.results = []
        
#     def split_into_detail_captions(self, text_long):
#         detail_caps = [p.strip() for p in text_long.split('.') if p.strip()]
#         return detail_caps

#     def extract_image_patches(self, images, patch_size=16):
#         B, C, H, W = images.shape
#         patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
#         patches = patches.permute(0, 2, 1, 3, 4)
#         return patches

#     def train_epoch(self, dataloader, epoch, start_iter=0):
#         running_loss = 0.0
#         running_loss_long = 0.0
#         running_loss_short = 0.0
#         running_loss_finegrained = 0.0
#         num_batches_per_epoch = len(dataloader)
        
#         for i, (images, texts, short_text) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", disable=(self.local_rank != 0))):
#             step = num_batches_per_epoch * epoch + i
#             if step < start_iter:
#                 continue
            
#             # Tokenize detail captions cho từng ảnh
#             tokenized_detail_caps = [
#                 [longclip.tokenize(dc, truncate=True).to(self.device) for dc in self.split_into_detail_captions(text)]
#                 for text in texts
#             ]
#             texts = longclip.tokenize(texts, truncate=True).to(self.device)
#             short_text = longclip.tokenize(short_text, truncate=True).to(self.device)
            
#             self.scheduler(step)
#             self.optimizer.zero_grad()
            
#             # Tách ảnh thành patch
#             image_patches = self.extract_image_patches(images, patch_size=16).to(self.device)
            
#             with torch.cuda.amp.autocast():
#                 # Lưu ý: khi dùng DDP, gọi forward vẫn như bình thường.
#                 # Nếu cần gọi các phương thức của model gốc, hãy dùng self.model.module
#                 loss_long, loss_short, finegrained_loss = self.model(images, texts, short_text, tokenized_detail_caps, image_patches)
#                 loss = loss_long + loss_short + finegrained_loss
                
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
            
#             if self.local_rank == 0:
#                 print(f"Epoch [{epoch}], Step [{step}], Total Loss: {loss.item():.4f}, "
#                       f"ITCL: {loss_long.item():.4f}, ITCS: {loss_short.item():.4f}, "
#                       f"Finegrained Loss: {finegrained_loss.item():.4f}")
            
#             running_loss += loss.item()
#             running_loss_long += loss_long.item()
#             running_loss_short += loss_short.item()
#             running_loss_finegrained += finegrained_loss.item()
        
#         avg_loss = running_loss / num_batches_per_epoch
#         avg_loss_long = running_loss_long / num_batches_per_epoch
#         avg_loss_short = running_loss_short / num_batches_per_epoch
#         avg_loss_finegrained = running_loss_finegrained / num_batches_per_epoch
        
#         if self.local_rank == 0:
#             print(f"========== Epoch {epoch} completed ==========")
#             print(f"Avg Total Loss: {avg_loss:.4f}, Avg ITCL: {avg_loss_long:.4f}, "
#                   f"Avg ITCS: {avg_loss_short:.4f}, Avg Finegrained Loss: {avg_loss_finegrained:.4f}")
#         return avg_loss, avg_loss_long, avg_loss_short, avg_loss_finegrained

#     @torch.no_grad()
#     def test_epoch(self, dataloader):
#         for id, (images, text) in enumerate(tqdm(dataloader, desc="Testing", disable=(self.local_rank != 0))):
#             images = images.to(self.device)
#             # Khi dùng DDP, gọi các hàm của model gốc qua self.model.module
#             image_features = self.model.module.encode_image(images)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#             text = longclip.tokenize(text, truncate=True).to(self.device)
#             text_feature = self.model.module.encode_text(text)
#             text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

#             correct = 0
#             total = 0
#             for i in range(text_feature.shape[0]):
#                 sim = text_feature[i] @ image_features.T
#                 correct_i = torch.argmax(sim)
#                 if i == correct_i:
#                     correct += 1
#                 total += 1
#         return correct / total

#     def test(self, epoch=0):
#         if self.local_rank == 0:  # Chỉ process chính in kết quả test
#             self.model.eval()
#             testset = share4v_val_dataset()
#             test_sampler = DistributedSampler(testset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
#             testloader = DataLoader(testset, batch_size=1000, num_workers=32, pin_memory=True, sampler=test_sampler)
#             with torch.no_grad():
#                 acc = self.test_epoch(testloader)
#                 print("=====================================")
#                 print(f"Test mean of share4v retrieval: {acc}")
#                 print("=====================================")
#         return

#     def train(self, resume=False, warmup_length=200):
#         trainset = share4v_train_dataset()
#         train_sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
#         train_loader = DataLoader(trainset, batch_size=self.batch_size, num_workers=32, pin_memory=True, sampler=train_sampler)
#         self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length,
#                                    steps=self.num_epoch * len(train_loader))
#         start_epoch = 0
#         resume_iter = 0
        
#         for epoch in range(start_epoch, self.num_epoch):
#             train_sampler.set_epoch(epoch)  # Đảm bảo shuffle đồng bộ giữa các epoch
#             avg_loss, avg_loss_long, avg_loss_short, avg_loss_finegrained = self.train_epoch(train_loader, epoch, start_iter=resume_iter)
            
#             if self.local_rank == 0:
#                 now = datetime.now()
#                 formatted_date = now.strftime("%m-%d--%H_%M_%S_")
#                 checkpoint_name = f"Propose-longclip.pt"
#                 torch.save(self.model.module.state_dict(), os.path.join(self.ckptdir, checkpoint_name))
                
#                 try:
#                     eval_output = subprocess.check_output(
#                         ["python", "imagenet.py"],
#                         cwd="../eval/classification/imagenet",
#                         stderr=subprocess.STDOUT
#                     )
#                     eval_result = eval_output.decode("utf-8")
#                     print("========== Evaluation Result ==========")
#                     print(eval_result)
#                     print("========================================")
#                 except subprocess.CalledProcessError as e:
#                     eval_result = "Error"
#                     print("Error running the evaluation script:")
#                     print(e.output.decode("utf-8"))
                
#                 self.results.append({
#                     "epoch": epoch,
#                     "avg_total_loss": avg_loss,
#                     "avg_loss_long": avg_loss_long,
#                     "avg_loss_short": avg_loss_short,
#                     "avg_finegrained_loss": avg_loss_finegrained,
#                     "eval_result": eval_result.strip()
#                 })
                
#                 results_df = pd.DataFrame(self.results)
#                 excel_file_path = os.path.join(self.logdir, "epoch_results.xlsx")
#                 results_df.to_excel(excel_file_path, index=False)
        
#         # Sau khi huấn luyện xong, tất cả các process đóng process group
#         dist.destroy_process_group()


# def setup_distributed(backend="nccl", port=None):
#     """Initialize distributed training environment.
#     support both slurm and torch.distributed.launch
#     see torch.distributed.init_process_group() for more details
#     """
#     num_gpus = torch.cuda.device_count()

#     if "SLURM_JOB_ID" in os.environ:
#         rank = int(os.environ["SLURM_PROCID"])
#         world_size = int(os.environ["SLURM_NTASKS"])
#         node_list = os.environ["SLURM_NODELIST"]
#         addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
#         # specify master port
#         if port is not None:
#             os.environ["MASTER_PORT"] = str(port)
#         elif "MASTER_PORT" not in os.environ:
#             os.environ["MASTER_PORT"] = "29522"
#         if "MASTER_ADDR" not in os.environ:
#             os.environ["MASTER_ADDR"] = addr
#         os.environ["WORLD_SIZE"] = str(world_size)
#         os.environ["LOCAL_RANK"] = str(rank % num_gpus)
#         os.environ["RANK"] = str(rank)
#     else:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])

#     torch.cuda.set_device(rank % num_gpus)
    
#     dist.init_process_group(
#         backend=backend,
#         world_size=world_size,
#         rank=rank,
#     )
#     torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
#     return rank, rank % num_gpus



# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='params')
#     parser.add_argument('--lr', default=1e-6, type=float, help='Learning rate.')
#     parser.add_argument('--weight_decay', default=1e-2, type=float, help='Weight decay.')
#     parser.add_argument('--log_scale', default=4.6052, type=float, help='CLIP temperature log scale.')
#     parser.add_argument("--exp_name", default="auto", type=str, help="Experiment name.")
#     parser.add_argument("--warmup_length", default=200, type=int, help="Warmup length.")
#     parser.add_argument("--base_model", default="ViT-B/16", help="CLIP Base Model")
#     parser.add_argument("--batch-size", type=int, default=50, help="Batch size per GPU.")
#     parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
#     parser.add_argument("--resume", default=False, action='store_true', help="Resume training from checkpoint.")
#     parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
#     # THÊM đối số local_rank để DDP hoạt động đúng
#     parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    
#     args = parser.parse_args()
#     print("DDP Done")
    
#     trainer = CLIP_Clean_Train(args=args)
#     trainer.train(resume=args.resume, warmup_length=args.warmup_length)

# ===============================================

# import torch
# #from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
# #the original concat_all_gather is abandoned because of no gradient backward
# from utils import is_dist_avail_and_initialized, accuracy
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from tqdm import tqdm

# import sys
# sys.path.append("..")

# from sharegpt4v import share4v_val_dataset, share4v_train_dataset
# from model import longclip

# from torch.utils.data.distributed import DistributedSampler
# from scheduler import cosine_lr
# import argparse
# import os
# import subprocess
# import collections
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# from datetime import datetime
# from torch.cuda.amp import GradScaler
# # import warnings
# # warnings.filterwarnings("ignore")
# import pandas as pd  # For saving results to Excel

# class CLIP_Clean_Train():
#     def __init__(self, args):
#         self.base_model = args.base_model
#         self.model, _ = longclip.load_from_clip(self.base_model, device='cpu', download_root=args.download_root)
#         self.model.train()
#         self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
        
#         # Chuyển model sang GPU
#         self.model = self.model.cuda()
#         # Nếu có nhiều GPU thì sử dụng DataParallel để phân phối công việc
#         # if torch.cuda.device_count() > 1:
#         #     print("Using", torch.cuda.device_count(), "GPUs")
#         #     self.model = torch.nn.DataParallel(self.model)
        
#         self.batch_size = args.batch_size
#         self.num_epoch = args.epochs
#         self.lr = args.lr
#         self.weight_decay = args.weight_decay
#         self.warmup_length = args.warmup_length
        
#         if args.exp_name == "auto":
#             self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
#         else:
#             self.logdir = args.exp_name
#         self.ckptdir = os.path.join(self.logdir, "ckpt")
#         os.makedirs(self.ckptdir, exist_ok=True)
#         self.writer = SummaryWriter(self.logdir)

#         self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         self.scaler = GradScaler()

#         # Danh sách lưu lại kết quả của mỗi epoch (loss và eval)
#         self.results = []
        
#     #TODO: viết thêm 1 hàm lấy 2 thằng ngẫu nhiên trong list detail cap (sample theo seed)
#     def split_into_detail_captions(self, text_long):
#         """
#         Nhận vào chuỗi long caption và tách thành danh sách các detail caption.
#         Ví dụ:
#           Input: "A cat sitting on a mat. The cat is playing with a ball. Clear sky."
#           Output: ["A cat sitting on a mat", "The cat is playing with a ball", "Clear sky"]
#         """
#         detail_caps = [p.strip() for p in text_long.split('.') if p.strip()]
#         return detail_caps
    
#     def extract_image_patches(self, images, patch_size=16):
#         """
#         Tách ảnh thành các patch.
#         Args:
#             images: Tensor có shape [B, C, H, W]
#             patch_size: Kích thước patch
#         Returns:
#             patches: Tensor có shape [B, num_patches, C, patch_size, patch_size]
#         """
#         B, C, H, W = images.shape
#         patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
#         patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
#         patches = patches.permute(0, 2, 1, 3, 4)
#         return patches

#     def train_epoch(self, dataloader, epoch, start_iter=0):
#         running_loss = 0.0
#         running_loss_long = 0.0
#         running_loss_short = 0.0
#         running_loss_finegrained = 0.0
#         running_loss_I2mT_loss = 0.0
#         num_batches_per_epoch = len(dataloader)
        
#         for i, (images, texts, short_text) in enumerate(tqdm(dataloader)):
#             step = num_batches_per_epoch * epoch + i
#             if step < start_iter:
#                 continue
            
#             # Tokenize các detail caption cho từng ảnh
#             tokenized_detail_caps = [
#                 [longclip.tokenize(dc, truncate=True).cuda() for dc in self.split_into_detail_captions(text)]
#                 for text in texts
#             ]

#             import random

#             # Đặt seed cố định để đảm bảo kết quả tái lập
#             seed_value = 42
#             random.seed(seed_value)

#             # tokenized_detail_caps_sample = [
#             #     random.sample(
#             #         [longclip.tokenize(dc, truncate=True).cuda() for dc in self.split_into_detail_captions(text)],
#             #         2
#             #     )
#             #     for text in texts
#             # ]

#             texts_token = longclip.tokenize(texts, truncate=True).cuda()
#             short_text = longclip.tokenize(short_text, truncate=True).cuda()

#             seed_value = 42
#             random.seed(seed_value)

#             tokenized_detail_caps_sample = []
#             for text in texts:
#                 # Tokenize và chuyển sang cuda cho mỗi detail caption của text
#                 detail_caps = [longclip.tokenize(dc, truncate=True).cuda() for dc in self.split_into_detail_captions(text)]
                
#                 # Nếu số lượng detail caption ít hơn 2, nhân đôi các phần tử có sẵn
#                 if len(detail_caps) < 2:
#                     sample = detail_caps * 2  # Duplicate: nếu chỉ có 1 phần tử, sau khi nhân đôi sẽ có 2 phần tử
#                 else:
#                     sample = random.sample(detail_caps, 2)

#                 # sample.extend([texts_token, short_text])
                
#                 tokenized_detail_caps_sample.append(sample)

            
#             self.scheduler(step)
#             self.optimizer.zero_grad()

#             # Tách ảnh thành patch
#             image_patches = self.extract_image_patches(images, patch_size=16).cuda()
            
#             with torch.cuda.amp.autocast():
#                 # Gọi forward của model để tính các loss, note rằng DataParallel tự chia batch cho các GPU
#                 finegrained_loss, I2mT_loss = self.model(images, texts_token, short_text, tokenized_detail_caps, tokenized_detail_caps_sample)
#                 loss = finegrained_loss + I2mT_loss
                
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
            
#             print(f"Epoch [{epoch}], Step [{step}], Total Loss: {loss.item():.4f}, "
#                   f"finegrained_loss: {finegrained_loss.item():.4f}, "
#                   f"I2mT_loss: {I2mT_loss.item():.4f}")
            
#             running_loss += loss.item()
#             running_loss_finegrained += finegrained_loss.item()
#             running_loss_I2mT_loss += I2mT_loss.item()
        
#         avg_loss = running_loss / num_batches_per_epoch
#         avg_loss_finegrained = running_loss_finegrained / num_batches_per_epoch
#         avg_loss_I2mT_loss = running_loss_I2mT_loss / num_batches_per_epoch
        
#         print(f"========== Epoch {epoch} completed ==========")
#         print(f"Average Total Loss: {avg_loss:.4f}, "
#               f"Average Finegrained Loss: {avg_loss_finegrained:.4f}, "
#               f"Average I2mT_loss: {avg_loss_I2mT_loss:.4f}")
#         return avg_loss, avg_loss_finegrained, avg_loss_I2mT_loss

#     @torch.no_grad()
#     def test_epoch(self, dataloader):
#         # Lưu ý: vì dùng DataParallel, truy cập các hàm encode qua self.model.module
#         for id, (images, text) in enumerate(tqdm(dataloader)):
#             images = images.cuda()
#             image_features = self.model.module.encode_image(images)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)

#             text = longclip.tokenize(text, truncate=True).cuda()
#             text_feature = self.model.module.encode_text(text)
#             text_feature /= text_feature.norm(dim=-1, keepdim=True)

#             correct = 0
#             total = 0
#             for i in range(text_feature.shape[0]):
#                 sim = text_feature[i] @ image_features.T
#                 sim = sim.squeeze()
#                 correct_i = torch.argmax(sim)
#                 if i == correct_i:
#                     correct += 1
#                 total += 1

#         return correct / total
    
#     def test(self, epoch=0):
#         rank = 0
#         if rank == 0:
#             self.model.eval()
#             testset = share4v_val_dataset()
#             testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=32, pin_memory=True)
#             with torch.no_grad():
#                 acc = self.test_epoch(testloader)
#                 print("=====================================")
#                 print(f"Test mean of share4v retrieval: {acc}")
#                 print("=====================================")
#             return
    
#     def train(self, resume=False, warmup_length=200):
#         trainset = share4v_train_dataset()
#         train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, num_workers=48, pin_memory=True)
#         self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length,
#                                    steps=self.num_epoch * len(train_loader))
#         start_epoch = 0
#         resume_iter = 0
        
#         for epoch in range(start_epoch, self.num_epoch):
#             avg_loss, avg_loss_finegrained, avg_loss_I2mT_loss = self.train_epoch(train_loader, epoch, start_iter=resume_iter)
            
#             now = datetime.now()
#             formatted_date = now.strftime("%m-%d--%H_%M_%S_")
#             checkpoint_name = f"Propose-b16-longclip-w05-{formatted_date}-mul_only_grain_i2mt_optz.pt"
#             torch.save(self.model.state_dict(), os.path.join(self.ckptdir, checkpoint_name))
            
#             try:
#                 eval_output = subprocess.check_output(
#                     ["python", "imagenet.py"],
#                     cwd="../eval/classification/imagenet",
#                     stderr=subprocess.STDOUT
#                 )
#                 eval_result = eval_output.decode("utf-8").strip().splitlines()[-1]
#                 print("========== Evaluation Result ==========")
#                 print(eval_result)
#                 print("========================================")
#             except subprocess.CalledProcessError as e:
#                 eval_result = "Error"
#                 print("Error running the evaluation script:")
#                 print(e.output.decode("utf-8"))
            
#             self.results.append({
#                 "epoch": epoch,
#                 "avg_total_loss": avg_loss,
#                 "avg_finegrained_loss": avg_loss_finegrained,
#                 "avg_loss_I2mT_loss": avg_loss_I2mT_loss,
#                 "eval_result": eval_result.strip()
#             })
            
#             results_df = pd.DataFrame(self.results)
#             excel_file_path = os.path.join(self.logdir, "epoch_results_b16_w05_mul_only_grain_i2mt_optz.xlsx")
#             results_df.to_excel(excel_file_path, index=False)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='params')
#     parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
#     parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
#     parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
#     parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
#     parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
#     # parser.add_argument("--base_model", default="ViT-L/14", help="CLIP Base Model")
#     parser.add_argument("--base_model", default="ViT-B/16", help="CLIP Base Model")
#     # parser.add_argument(
#     #     "--batch-size", type=int, default=128, help="Batch size per gpu."#112
#     # )
#     parser.add_argument(
#         "--batch-size", type=int, default=16, help="Batch size per gpu."#112
#     )
#     parser.add_argument(
#         "--epochs", type=int, default=3, help="Number of epochs to train for."
#     )
#     parser.add_argument(
#         "--resume",
#         default=False,
#         action='store_true',
#         help="resume training from checkpoint."
#     )
#     parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
#     args = parser.parse_args()
#     # rank,local_rank = setup_distributed()
#     print("DDP Done")

#     trainer = CLIP_Clean_Train(
#         # rank=rank,
#         # local_rank=local_rank, 
#         args=args
#         )
#     trainer.train(resume=args.resume, warmup_length=args.warmup_length)