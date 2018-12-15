from torchvision import models

def get_model():
    '''Obtains the features of a pretrained model.
    
    VGG11 network trained on ImageNet.
    '''
    model = models.vgg11(pretrained=True)
    return model.features