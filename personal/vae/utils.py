from torchvision import transforms


class ApplyPad:
    def __call__(self, img):
        img = transforms.functional.pad(img, (2,2,2,2), fill=0)
        return img
            