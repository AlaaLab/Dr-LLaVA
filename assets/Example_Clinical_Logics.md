# Clinical Logics for Analyzing ChestX-ray8 Dataset

The ChestX-ray8 dataset contains X-ray images of the chest. To analyze these images and categorize them based on abnormalities, the following clinical logics can be applied:

## Initial Assessment

1. Determine if the chest X-ray shows any signs of abnormality.
    - If 'No Findings': The image is categorized as normal.
    - If abnormalities are present, proceed to Step 2.

## Disease Identification

2. Analyze the nature of the abnormalities in the X-ray image.
    - If the abnormalities are consistent with COVID-19 characteristics (e.g., ground-glass opacities, bilateral and peripheral lung involvement), the image is labeled as 'COVID-19'.
    - If the abnormalities are consistent with typical pneumonia characteristics (e.g., localized consolidation, bronchopneumonia patterns), the image is labeled as 'Pneumonia'.

## Final Categorization

3. Based on the analysis, categorize each chest X-ray image into one of the following:
    - COVID-19
    - Pneumonia
    - No Findings

