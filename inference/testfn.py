from signjoey.inference import inference



cfg_file = 'configs/sign_dcba_inf.yaml'

ckpt = None

result = inference(cfg_file=cfg_file, ckpt=ckpt)

print(result)