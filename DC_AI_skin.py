import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2
import efficientnet
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from torch import nn
import torch

st.write("""
# Multiple Classification by Supplementing Lesion Region Information on Lesion of Facial Skin using Deep Learning""")

PRETRAINED_ROOT = '/weights/'
EXTERNAL_DEPENDENCIES = {
    "Main": {
        "url": PRETRAINED_ROOT+"main_meta_loc.pth.tar",
        "num_class": 27
    },
    "Group1": {
        "url": PRETRAINED_ROOT+"group_01.pth.tar",
        "num_class": 2
    },
    "Group2": {
        "url": PRETRAINED_ROOT+"group_02.pth.tar",
        "num_class": 2
    },
    "Group3": {
        "url": PRETRAINED_ROOT+"group_03.pth.tar",
        "num_class": 8
    },
    "Group4": {
        "url": PRETRAINED_ROOT+"group_04.pth.tar",
        "num_class": 8
    },
    "Group5": {
        "url": PRETRAINED_ROOT+"group_05.pth.tar",
        "num_class": 2
    },
    "Group6": {
        "url": PRETRAINED_ROOT+"group_06.pth.tar",
        "num_class": 2
    },                    
}
MAIN_LIST = ['AK', 'BCC', "Bowen's disease", 'LP', 'Ota nevus', 
            'SCC', 'SK', 'acne', 'blue nevus', 'congenital melanocytic nevus', 
            'eczema', 'epidermal cyst', 'flammeus nevus' ,'hyperplasia', 'intradermal nevus',
            'lentigo', 'melanocytic nevus', 'melasma', 'milium', 'mucocele', 
            'nevus sebaceous', 'pilomatricoma', 'rosacea', 'syringoma', 
            'venous lake', 'verruca','vitiligo']

GROUP1 = ['BCC', 'SK']          # 1, 6
GROUP2 = ['SCC', 'SK']          # 5, 6
GROUP5 = ['AK', 'eczema']       # 0, 10
GROUP6 = ['acne', 'rosacea']    # 7, 22

GROUP3 = ["BCC",
         "blue nevus",                      # 8
         "congenital melanocytic nevus",    # 9
         "intradermal nevus",               # 14
         "melanocytic nevus",               # 16
         "flammeus nevus",                  # 12
         "nevus sebaceous",                 # 20
         "Ota nevus"]                       # 4

GROUP4 = ["SK",
         "blue nevus",                      # 8
         "congenital melanocytic nevus",    # 9
         "intradermal nevus",               # 14
         "melanocytic nevus",               # 16
         "flammeus nevus",                  # 12
         "nevus sebaceous",                 # 20
         "Ota nevus"]                       # 4


def main():
    st.title("Image Classification ")
    st.header("Lesion of Facial Skin Classification Example")
    st.text("Upload a skin Image for image classification")
    st.sidebar.header('Model')
    name = st.sidebar.selectbox('Name', ['Main', 'Group1', 'Group2','Group3','Group4','Group5','Group6'])
    st.sidebar.header('Metadata')
    age = st.sidebar.number_input('Age', min_value=0, max_value=100)
    sex = st.sidebar.number_input('Sex', min_value=0, max_value=1)
    st.sidebar.text('*0: male, 1: female')
    location = st.sidebar.number_input('Location', min_value=0, max_value=7)    
    st.sidebar.text('*0: Auricular, 1: Cheek, 2: Eyelid,')
    st.sidebar.text(' 3: Forehead, 4: Lip, 5: Mental,')
    st.sidebar.text(' 6: Nasal, 7: Neck')
    metadata = {'age':age, 'sex':sex, 'location':location}
    
    uploaded_file = st.file_uploader("Choose a skin lesion image ...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image, torch_image = load_image(uploaded_file)        
        
        model = load_network(name)
        st.image(image,use_column_width=True)
        efficient_net(torch_image, model, metadata, name)

def load_image(image):
    image = Image.open(image)
    image = image.convert('RGB')
    transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    torch_image = transform(image)
    torch_image = torch_image.unsqueeze(dim=0)

    return image, torch_image

@st.cache(allow_output_mutation=True)
def load_network(name):
    model = efficientnet.EfficientNet.from_pretrained('efficientnet-b0', num_classes=EXTERNAL_DEPENDENCIES[name]['num_class'])

    num_ftrs = model._fc.in_features    
    model._fc = nn.Linear(num_ftrs + EXTERNAL_DEPENDENCIES[name]['num_class'], EXTERNAL_DEPENDENCIES[name]['num_class'])
    model.emr_init()

    pretrained_dict = torch.load(EXTERNAL_DEPENDENCIES[name]['url'], map_location='cpu')
    try:
        model_dict = model.module.state_dict()
    except AttributeError:
        model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    return model

def run_model(model, image, emr):
    with torch.no_grad():
        outputs = model(image, emr)
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.numpy().squeeze()
        
        return outputs

def efficient_net(image, model, metadata, name):  
    age = torch.tensor([metadata['age']/100])
    sex = torch.tensor([metadata['sex']])
    location = torch.tensor([metadata['location']])
    emr = torch.stack((age, sex, location), 1)    
    if name == 'Main':
        outputs = run_model(model, image, emr)
        st.write('## Result: %s (%.2f)' % (MAIN_LIST[np.argmax(outputs)], outputs[np.argmax(outputs)]))
        st.bar_chart(outputs)
        st.text("0: AK, 1: BCC, 2: Bowen\'s disease, 3: LP, 4: Ota nevus, 5: SCC, 6: SK, 7: acne,")
        st.text("8: blue nevus, 9: congenital melanocytic nevus, 10: eczema, 11: epidermal cyst,")
        st.text("12: flammeus nevus ,13: hyperplasia, 14: intradermal nevus, 15: lentigo,")
        st.text("16: melanocytic nevus,17: melasma, 18: milium, 19: mucocele, 20: nevus sebaceous,")
        st.text("21: pilomatricoma, 22: rosacea, 23: syringoma,24: venous lake, 25: verruca, 26: vitiligo")

    elif name == 'Group1':    
        model = load_network(name)
        outputs = run_model(model, image, emr)
        st.write('## Result: %s (%.2f)' % (GROUP1[np.argmax(outputs)], outputs[np.argmax(outputs)]))
        st.bar_chart(outputs)
        
    elif name == 'Group2':    
        model = load_network(name)
        outputs = run_model(model, image, emr)
        st.write('## Result: %s (%.2f)' % (GROUP2[np.argmax(outputs)], outputs[np.argmax(outputs)]))
        st.bar_chart(outputs)

    elif name == 'Group3':    
        model = load_network(name)
        outputs = run_model(model, image, emr)
        st.write('## Result: %s (%.2f)' % (GROUP3[np.argmax(outputs)], outputs[np.argmax(outputs)]))
        st.bar_chart(outputs)

    elif name == 'Group4':    
        model = load_network(name)
        outputs = run_model(model, image, emr)
        st.write('## Result: %s (%.2f)' % (GROUP4[np.argmax(outputs)], outputs[np.argmax(outputs)]))
        st.bar_chart(outputs)

    elif name == 'Group5':    
        model = load_network(name)
        outputs = run_model(model, image, emr)
        st.write('## Result: %s (%.2f)' % (GROUP5[np.argmax(outputs)], outputs[np.argmax(outputs)]))
        st.bar_chart(outputs)

    elif name == 'Group6':    
        model = load_network(name)
        outputs = run_model(model, image, emr)
        st.write('## Result: %s (%.2f)' % (GROUP6[np.argmax(outputs)], outputs[np.argmax(outputs)]))
        st.bar_chart(outputs)


if __name__ == "__main__":
    main()