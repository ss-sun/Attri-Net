import torchxrayvision as xrv
import skimage, torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F


# Code Adapt from https://github.com/mlmed/gifsplanation/blob/main/gif-attributionmaps.ipynb

class gif_explainer():
    def __init__(self, model, labels):
        self.ae = xrv.autoencoders.ResNetAE(weights="101-elastic").cuda()
        self.model = model
        self.labels = labels

    def get_attributions(self, input, target_label_idx, positive_only=False):
        attr = self.compute_attribution(input, self.model, target_label_idx, ae=self.ae)
        return attr


    def thresholdf(self, x, percentile):
        return x * (x > np.percentile(x, percentile))


    def compute_attribution(self, image, model, target_label_idx, ret_params=False, fixrange=None, p=0.0, ae=None,
                            sigma=0, threshold=False):

        image = image.clone().detach()
        image_shape = image.shape[-2:]
        image = image * 1024

        def clean(saliency):
            saliency = np.abs(saliency)
            if sigma > 0:
                saliency = skimage.filters.gaussian(saliency,
                                                    mode='constant',
                                                    sigma=(sigma, sigma),
                                                    truncate=3.5)
            if threshold != False:
                saliency = self.thresholdf(saliency, 95 if threshold == True else threshold)
            return saliency

        method = "latentshift-max"
        if "latentshift" in method:

            z = ae.encode(image).detach()
            z.requires_grad = True
            xp = ae.decode(z, image_shape)
            # pred = F.sigmoid(model((image * p + xp * (1 - p))))[:, model.pathologies.index(target)]
            pred = F.sigmoid(model((image * p + xp * (1 - p))))[:, target_label_idx]
            dzdxp = torch.autograd.grad((pred), z)[0]

            cache = {}

            def compute_shift(lam):
                # print(lam)
                if lam not in cache:
                    xpp = ae.decode(z + dzdxp * lam, image_shape).detach()
                    # pred1 = F.sigmoid(model((image * p + xpp * (1 - p))))[:,
                    #         model.pathologies.index(target)].detach().cpu().numpy()
                    pred1 = F.sigmoid(model((image * p + xpp * (1 - p))))[:,
                            target_label_idx].detach().cpu().numpy()

                    cache[lam] = xpp, pred1
                return cache[lam]

            # determine range
            _, initial_pred = compute_shift(0)

            if fixrange:
                lbound, rbound = fixrange
            else:
                # search params
                step = 10
                # left range
                lbound = 0
                last_pred = initial_pred
                while True:
                    xpp, cur_pred = compute_shift(lbound)
                    # print("lbound",lbound, "last_pred",last_pred, "cur_pred",cur_pred)
                    if last_pred < cur_pred:
                        break
                    if initial_pred - 0.15 > cur_pred:
                        break
                    if lbound <= -1000:
                        break
                    last_pred = cur_pred
                    if np.abs(lbound) < step:
                        lbound = lbound - 1
                    else:
                        lbound = lbound - step

                # right range
                rbound = 0

            print(initial_pred, lbound, rbound)
            lambdas = np.arange(lbound, rbound, np.abs((lbound - rbound) / 10))
            y = []
            dimgs = []
            xp = ae.decode(z, image_shape)[0][0].unsqueeze(0).unsqueeze(0).detach()

            for lam in lambdas:
                xpp, pred = compute_shift(lam)
                dimgs.append(xpp.cpu().numpy())
                y.append(pred)

            if ret_params:
                params = {}
                params["dimgs"] = dimgs
                params["lambdas"] = lambdas
                params["y"] = y
                params["initial_pred"] = initial_pred
                return params


            if "-max" in method:
                dimage = np.max(np.abs(xp.cpu().numpy()[0][0] - dimgs[0][0]), 0)
            elif "-mean" in method:
                dimage = np.mean(np.abs(xp.cpu().numpy()[0][0] - dimgs[0][0]), 0)
            elif "-mm" in method:
                dimage = np.abs(dimgs[0][0][0] - dimgs[-1][0][0])
            elif "-int" in method:
                dimages = []
                for i in range(len(dimgs) - 1):
                    dimages.append(np.abs(dimgs[i][0][0] - dimgs[i + 1][0][0]))
                dimage = np.mean(dimages, 0)
            else:
                raise Exception("Unknown mode")

            dimage = clean(dimage)
            return dimage

