#!/usr/bin/env python3
from pypdf import PdfReader, PdfWriter
import os

pdf_path = "/home/antony/pdf/py_ml_w_py_skl.pdf"
output_dir = "/home/antony/pdf/chapters"

os.makedirs(output_dir, exist_ok=True)

# Chapter definitions: (name, start_page, end_page) - 1-indexed, inclusive
chapters = [
    ("Chapter_01_Giving_Computers_the_Ability_to_Learn_from_Data", 32, 49),
    ("Chapter_02_Training_Simple_Machine_Learning_Algorithms_for_Classification", 50, 83),
    ("Chapter_03_A_Tour_of_Machine_Learning_Classifiers_Using_Scikit-Learn", 84, 135),
    ("Chapter_04_Building_Good_Training_Datasets_Data_Preprocessing", 136, 169),
    ("Chapter_05_Compressing_Data_via_Dimensionality_Reduction", 170, 201),
    ("Chapter_06_Learning_Best_Practices_for_Model_Evaluation_and_Hyperparameter_Tuning", 202, 235),
    ("Chapter_07_Combining_Different_Models_for_Ensemble_Learning", 236, 277),
    ("Chapter_08_Applying_Machine_Learning_to_Sentiment_Analysis", 278, 299),
    ("Chapter_09_Predicting_Continuous_Target_Variables_with_Regression_Analysis", 300, 335),
    ("Chapter_10_Working_with_Unlabeled_Data_Clustering_Analysis", 336, 365),
    ("Chapter_11_Implementing_a_Multilayer_Artificial_Neural_Network_from_Scratch", 366, 399),
    ("Chapter_12_Parallelizing_Neural_Network_Training_with_PyTorch", 400, 439),
    ("Chapter_13_Going_Deeper_The_Mechanics_of_PyTorch", 440, 481),
    ("Chapter_14_Classifying_Images_with_Deep_Convolutional_Neural_Networks", 482, 529),
    ("Chapter_15_Modeling_Sequential_Data_Using_Recurrent_Neural_Networks", 530, 569),
    ("Chapter_16_Transformers_Improving_Natural_Language_Processing_with_Attention_Mechanisms", 570, 619),
    ("Chapter_17_Generative_Adversarial_Networks_for_Synthesizing_New_Data", 620, 667),
    ("Chapter_18_Graph_Neural_Networks_for_Capturing_Dependencies_in_Graph_Structured_Data", 668, 703),
    ("Chapter_19_Reinforcement_Learning_for_Decision_Making_in_Complex_Environments", 704, 749),
]

reader = PdfReader(pdf_path)
total_pages = len(reader.pages)

print(f"Total pages in PDF: {total_pages}")
print(f"Splitting into {len(chapters)} chapter files...\n")

for name, start, end in chapters:
    writer = PdfWriter()
    # Convert from 1-indexed to 0-indexed
    for page_num in range(start - 1, end):
        writer.add_page(reader.pages[page_num])

    output_path = os.path.join(output_dir, f"{name}.pdf")
    with open(output_path, "wb") as output_file:
        writer.write(output_file)

    print(f"Created: {name}.pdf (pages {start}-{end}, {end - start + 1} pages)")

print(f"\nDone! All chapters saved to: {output_dir}")
