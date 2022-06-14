# Disentangling visual and written concepts in CLIP 
#### Official PyTorch implementation of the CVPR 2022 oral paper, [project website](https://joaanna.github.io/disentangling_spelling_in_clip/)


- [ ] training projection script
- [ ] uploaded pre-trained models
- [ ] text-to-image generation notebook
- [x] Typographic Attack Dataset released


---
## Applications
To use the pre-trained models:

`
W = torch.load(clip_subspace_path)['W'].data.cpu().numpy()

def project(x, W):
    x_projected = x @ W
    x_projected /= np.linalg.norm(x_projected, axis=1)[:, None]
    return x_projected`

### Text-to-Image generation
For generating images we refer to the notebook of Kathrine Crowson https://colab.research.google.com/drive/15UwYDsnNeldJFHJ9NdgYBYeo6xPmSelP. To generate images using the projection matrices, we compute the objective function using the projected features instead of CLIP feature vectors. 
We plan to provide a modified notebook to show-case this application.
The OCR detection was done using `easyocr` toolbox. 

----------- 

## Typographic attack dataset
We collected 180 images of 20 objects with 8 typographic attack labels. The dataset is available at https://drive.google.com/drive/folders/17v74pmQs2zHrs9sJ_8D0bfrz5ykB5ZW2?usp=sharing. 
The dataset contains annotation json file `annotation.json` with format a list of files: `{"IMG_2934.JPG": {"true object": "cup", "typographic attack label": "good"}`.
----

### Citation
If you use this data, please cite the following papers:
`@inproceedings{materzynskadisentangling,
Author = {Joanna Materzynska and Antonio Torralba and David Bau},
Title = {Disentangling visual and written concepts in CLIP},
Year = {2022},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}`



