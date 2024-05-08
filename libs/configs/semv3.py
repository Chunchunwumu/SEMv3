import os
import torch
device = torch.device('cuda')

model_name = "Model_points_delta_conv_2_new_merger_v1" #
tips = model_name
# train dataset
train_lrcs_path = [
    "/train20/intern/permanent/zrzhang6/qcxia/TSR/Dataset/companydataset/output_dir/extract_available/train/train_v5.lrc"
]

# 12g
train_max_pixel_nums = 4000 * 4000
height_norm = 8
train_bucket_seps = (300, 300, 6, 6, 10000)
train_max_row_nums = 20
train_max_col_nums = 20
train_max_batch_size = 24
train_num_workers = 1
max_img_size = 1600
min_img_size = 64
train_rota=False

# valid dataset
valid_lrc_path = ["/train20/intern/permanent/zrzhang6/qcxia/TSR/Dataset/companydataset/output_dir/extract_available/valid/valid_with_line_v5.lrc"]

valid_num_workers = 0
valid_batch_size = 1
is_need_rect = False
is_need_rotaterect = False
iou_list =sorted([0.6])
type_list = sorted(["segmentation"]) #,"segmentation_rotaterect","segmentation_rect"
valid_rota=False

backbone=dict(
    type='ResNetV1d',
    depth=34,
    init_cfg=dict(type='Pretrained', checkpoint="./PretrainModel/ResNet/resnet_34.pth")
)

# neck
neck=dict(
    in_channels=[64, 128, 256, 512],
    out_channels=256
)

# posemb
posemb=dict(
    in_channels=256,
    init_size = 4096
)

# resa
use_resa = False

# row split head
row_split_head=dict(
    split_type='1d', # '1d' '2d' --> whether to use CE and CTC loss
    line_type='row',
    down_stride=16,
    resa=dict(
        iters=3,
        line_type='row',
        channel=256,
        spatial_kernel=5
    ),
    feature_branch_kernelSzie=1, 
    kernel_branch_kernelSize=1, 
    feature_branch_act=dict(type='ReLU'),
    in_channels=256,
    loss=dict(
        type='focal', # 'bce', 'focal'(default), 'dice'
        div='pos', # 'pos' 'all' --> divided by positive samples or all samples,
        factor=1.
    )
)

# col split head
col_split_head=dict(
    split_type='1d', # '1d' '2d' --> whether to use CE and CTC loss
    line_type='col',
    down_stride=16,
    resa=dict(
        iters=3,
        line_type='col',
        channel=256,
        spatial_kernel=5
    ),
    feature_branch_kernelSzie=1, 
    kernel_branch_kernelSize=1, 
    feature_branch_act=dict(type='ReLU'),
    in_channels=256,
    loss=dict(
        type='focal', # 'bce', 'focal', 'dice'
        div='pos', # 'pos' 'all' --> divided by positive samples or all samples,
        factor=1.
    )
)

# grid extractor
grid_extractor=dict(
    in_channels=256,
    out_channels=512,
    grid_type='bbox', # , 'obb', 'corner'
    pool_size=(3,3),
    scale=0.5,
    num_attention_layers=1,
    num_attention_heads=8,
    intermediate_size=1024, # unused parameter
    dropout_prob=0.1 # unused parameter
)

# merge head
merge_head=dict(
    in_channels=512,
    num_kernel_layers=3,
    loss=dict(
        type='focal', # 'bce', 'focal', 'dice'
        div='pos', # 'pos' 'all' --> divided by positive samples or all samples,
        factor=1.
    )
)

is_visualize = True
# train params
base_lr = 1e-4
min_lr = 1e-6
weight_decay = 0

num_epochs = 121
start_eval = 80
eval_epochs = 5
sync_rate = 20
log_sep = 100

config_name = os.path.basename(__file__).split('.')[0]
work_dir = f'/train20/intern/permanent/cxqin/TSR/code/semv3_open_source/experiments/debug/{config_name}'+"_"+tips
valid_vis_path = None
train_checkpoint = None
resume_path = None
eval_checkpoint = "/train20/intern/permanent/zrzhang6/qcxia/TSR/code/SEMv3/experiments/v1/config_IFLYTAB_delta_conv_2_new_merger_v1_bbox/2023_12_19_10_13_Model_points_delta_conv_2_new_merger_v1/best_cell_iou_f1_model.pth"
infer_checkpoint = "/train20/intern/permanent/zrzhang6/qcxia/TSR/code/SEMv3/experiments/v1/config_IFLYTAB_delta_conv_2_new_merger_v1_bbox/2023_12_19_10_13_Model_points_delta_conv_2_new_merger_v1/best_cell_iou_f1_model.pth"

