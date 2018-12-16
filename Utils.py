from torchvision import models, transforms
from PIL import Image

def get_model():
    '''Obtains the features of a pretrained model.
    
    VGG11 network trained on ImageNet.
    '''
    model = models.vgg11(pretrained=True)
    return model.features

def load_image(path):
    image = Image.open(path).convert('RGB')
    
    img_transform = transforms.Compose([
                        transforms.Resize(640),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
    image = img_transform(image)
    image = image[:3,:,:].unsqueeze(0)
    
    return image