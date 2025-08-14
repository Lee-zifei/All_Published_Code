from .wudt_STAnet import WUDT_STAnet
from .wudtnet import WUDTnet


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'WUDTnet':
        model = WUDTnet(img_size=config.DATA.IMG_SIZE,
                        in_chans=config.MODEL.DT.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.DT.EMBED_DIM,
                        depths=config.MODEL.DT.DEPTHS,
                        num_heads=config.MODEL.DT.NUM_HEADS,
                        window_size=config.MODEL.DT.WINDOW_SIZE,
                        mlp_ratio=2,
                        qkv_bias=config.MODEL.DT.QKV_BIAS,
                        qk_scale=config.MODEL.DT.QK_SCALE,
                        drop_rate=0.,
                        drop_path_rate=0.1,
                        patch_norm=config.MODEL.DT.PATCH_NORM, config=config)
    elif model_type == 'WUDT_STAnet':
        model = WUDT_STAnet(
                    # embed_dim=[96, 192, 448, 640], # 95M, 15.6G, 269 FPS
                    img_size=config.DATA.IMG_SIZE,
                    in_chans=config.MODEL.DT2.IN_CHANS,
                    num_classes=config.MODEL.NUM_CLASSES,
                    embed_dim=config.MODEL.DT2.EMBED_DIM,
                    depths=config.MODEL.DT2.DEPTHS ,
                    num_heads=config.MODEL.DT2.NUM_HEADS ,
                    n_iter=config.MODEL.DT2.NITER, 
                    stoken_size=config.MODEL.DT2.STOKEN_SIZE, # for 224/384
                    projection=1024,
                    mlp_ratio=config.MODEL.DT2.MLP_RATIO,
                    qkv_bias=config.MODEL.DT2.QKV_BIAS,
                    qk_scale=config.MODEL.DT2.QK_SCALE,
                    drop_rate=0,
                    drop_path_rate=0.6 , 
                    layerscale=[False, False, True],
                    init_values=1e-6,config=config)

    return model
