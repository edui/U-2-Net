from model import U2NET
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
"""

# convert trained model to mobile

from network import UNet as HUNet
model = HUNet(128)
pretrained_model_h = torch.load(
        'models/model_ep_48.pth.tar', map_location='cpu')

model.load_state_dict(pretrained_model_h["state_dict"])

model.eval()

example = torch.rand(1, 3, 218, 218)
traced_script_module = torch.jit.script(model, example)
traced_script_module.save("heightfinder.pt")
"""


def prepare_save(model, fused):
    model.to(torch.device("cpu"))
    model.eval()
    torchscript_model = torch.jit.script(model)

    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized,
                   "mobile_model.pt" if not fused else "mobile_model_fused.pt")


def post_prepare_save(model, fused):
    model.to(torch.device("cpu"))
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model.eval(), inplace=True)
    torchscript_model = torch.jit.script(model)

    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized,
                   "mobile_model.pt" if not fused else "mobile_model_fused.pt")


def qat_prepare_save(model, fused):
    model.train()
    # model.to(torch.device("cpu"))
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    model_qat = torch.quantization.prepare_qat(model, inplace=False)

    # calibration
    # training_loop(model_qat)
    model_qat.eval()

    # quantization aware training goes here
    model_qat = torch.quantization.convert(model_qat, inplace=False)
    torchscript_model = torch.jit.script(model_qat)

    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized,
                   "mobile_model.pt" if not fused else "mobile_model_fused.pt")


model = U2NET(3, 1)
pretrained_model_hw = torch.load(
    'model/u2net_human_seg.pth', map_location='cpu')
model.load_state_dict(pretrained_model_hw["state_dict"])
# print(model)
prepare_save(model, False)
#qat_prepare_save(model, False)

#torch.save(model.state_dict(), 'mogiz.pth')
