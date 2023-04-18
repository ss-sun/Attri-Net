import torch
from captum.attr import GuidedBackprop, LayerGradCam, LayerAttribution
from train_utils import to_numpy



class GCam_explainer():
    def __init__(self, model, labels):
        self.explainer = LayerGradCam(model, model.layer4[2].conv3)
        self.model = model
        self.labels = labels

    def get_attributions(self, input, target_label_idx, positive_only):
        attributions_lgc = self.explainer.attribute(input, target=target_label_idx)
        upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input.shape[2:])
        if positive_only:
            upsamp_attr_lgc = upsamp_attr_lgc.clamp(0)
        return to_numpy(upsamp_attr_lgc).squeeze()


class GB_explainer():
    def __init__(self, model, labels):
        self.explainer = GuidedBackprop(model)
        self.model = model
        self.labels = labels
        self.replace_relu(self.model, 'model')

    def get_attributions(self, input, target_label_idx, positive_only):

        attr = self.explainer.attribute(input, target=target_label_idx)
        if positive_only:
            attr = attr.clamp(0)
        return to_numpy(attr).squeeze()

    def replace_relu(self, module, name):
        # Go through all attributes of module nn.module (e.g. network or layer) and set inplace to False.
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == torch.nn.ReLU:
                print('replaced: ', name, attr_str)
                new_relu = torch.nn.ReLU(inplace=False)
                setattr(module, attr_str, new_relu)
        # Iterate through immediate child modules.
        for name, immediate_child_module in module.named_children():
            self.replace_relu(immediate_child_module, name)


