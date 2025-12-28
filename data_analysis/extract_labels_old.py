# This is the new data processing method based on new standard introduced after the meeting in Germany.
# For more information please refer to folder: SA-CVD/Final-Discrepencies-Resolved

import os
import mne
import argparse
import numpy as np
import pandas as pd
import datetime
import neurokit2 as nk
import pdfplumber
import cv2
import re
# from easyocr import Reader

# reader = Reader(['en'], gpu=True)

dataset_dir = r"/data/R.Janini_Work/SA-CVD/dataset"
edf_dir = os.path.join(dataset_dir, "Embla")
meta_data_dir = os.path.join(dataset_dir, "Meta data")

CARDIOVASCULAR_AREA = 403767
SLEEP_DISORDER_AREA = 308448

APNEA_EVENTS_NAMES = ["APNEA", "APNEA-CENTRAL", "APNEA-MIXED", "APNEA-OBSTRUCTIVE"]
HYPOPNEA_EVENTS_NAMES = ["HYPOPNEA", "HYPOPNEA-CENTRAL", "HYPOPNEA-MIXED", "HYPOPNEA-OBSTRUCTIVE"]

def reverse_search(base_dir, edf_test):
    name, id = edf_test.split(" ")
    id = int(id[1]) - 1
    new_name = f"{name} ({id})"
   
    file_variations = [
                    f"Follow Up {id}.pdf",
                    f"Follow-Up {id}.pdf",
                    f"Follow up {id}.pdf",
                    f"Follow-up {id}.pdf",
                ]

    meta_data_file = next(
                    (os.path.join(base_dir, file) for file in file_variations if os.path.exists(os.path.join(base_dir, file))),
                    None
                ) 
                
    if meta_data_file is None:
        if id < 0:
            meta_data_file = os.path.join(base_dir, "Baseline.pdf")
            assert os.path.exists(meta_data_file), f"Baseline file {meta_data_file} does not exist"
            return meta_data_file
        else:
            return reverse_search(base_dir, new_name)
    else:
        return meta_data_file

def find_AHI(reader, pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        rects = []
        num_pages = len(pdf.pages)

        page = pdf.pages[2]
        pil_image = page.to_image(resolution=300).original
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

        binary = cv2.adaptiveThreshold(
                cv2_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
        
        text = reader.readtext(binary)  
        text = " ".join([result[1] for result in text])
        match = re.search(r"Apnea Hypopnea Index \(AHI\):.*?(\d+\.\d+)", text)
        ahi_value = match.group(1) if match else 100000
        
        return ahi_value
                    


def export_meta_data(LOOKFOR):
    cvd_wo_apnea = 0
    cvd_with_apnea = 0

    no_cvd_wo_apnea = 0
    no_cvd_w_apnea = 0


    export = 0
    if os.path.exists("/data/R.Janini_Work/SA-CVD/dataset/meta_data_v3.csv"):
        meta_data_exported = pd.read_csv("/data/R.Janini_Work/SA-CVD/dataset/meta_data_v3.csv")
        print("Meta data file already exists")
        return meta_data_exported
    else:
        export = 1
        from easyocr import Reader
        from ocr import checkbox_analysis

        reader = Reader(["en"], gpu=True)
        meta_data_exported = {"edf_file_path" : [],
                            "annotation_file_path": [],
                            "meta_data_file_path": [],
                            "apnea_checkboxes": [],
                            "cvd_checkboxes": [],
                            "AHI": [],
                            "has_apnea": [],
                            "has_cvd": []}
        
    for test_year in os.listdir(edf_dir):
        for edf_test in os.listdir(os.path.join(edf_dir, test_year)):
            edf_file_path = os.path.join(
                edf_dir, test_year, edf_test, edf_test + ".edf"
            )
            ann_file_path = os.path.join(edf_dir, test_year, edf_test, "xx.txt")
            assert os.path.exists(ann_file_path), f"File {ann_file_path} does not exist"

            if " " not in edf_test: meta_data_file = os.path.join(meta_data_dir, edf_test, "Baseline.pdf")
            else:
                id = edf_test.split(" ")[1][1]
                base_dir = os.path.join(meta_data_dir, edf_test.split(" ")[0])
                file_variations = [
                    f"Follow Up {id}.pdf",
                    f"Follow-Up {id}.pdf",
                    f"Follow up {id}.pdf",
                    f"Follow-up {id}.pdf",
                ]

                meta_data_file = next(
                    (os.path.join(base_dir, file) for file in file_variations if os.path.exists(os.path.join(base_dir, file))),
                    None
                )
                
                if meta_data_file is None:
                    meta_data_file = reverse_search(base_dir, edf_test)
                assert os.path.exists(meta_data_file), f"File {meta_data_file} does not exist"

                if export == 1:
                    AHI = find_AHI(reader, meta_data_file)
                    apnea_checkboxes = checkbox_analysis(meta_data_file, SLEEP_DISORDER_AREA, reader)
                    cvd_checkboxes = checkbox_analysis(meta_data_file, CARDIOVASCULAR_AREA, reader)

                    if AHI == 100000:
                        has_apnea = "idk"
                    else:
                        has_apnea = "yes" if float(AHI) > 5 else "no"

                    
                    if len(cvd_checkboxes) == 0:
                        has_cvd = "no"
                    elif (has_apnea == 'idk' or has_apnea == 'no') and len(cvd_checkboxes) > 0:
                        has_cvd = "idk"
                    else:
                        has_cvd = "yes"

                    meta_data_exported["edf_file_path"].append(edf_file_path)
                    meta_data_exported["annotation_file_path"].append(ann_file_path)
                    meta_data_exported["meta_data_file_path"].append(meta_data_file)
                    meta_data_exported['apnea_checkboxes'].append("; ".join(apnea_checkboxes))
                    meta_data_exported['cvd_checkboxes'].append("; ".join(cvd_checkboxes))
                    meta_data_exported['AHI'].append(AHI)
                    meta_data_exported['has_apnea'].append(has_apnea)
                    meta_data_exported['has_cvd'].append(has_cvd)

                    print("-----------------------------------")
                    print("Edf file path: ", edf_file_path)
                    print("Annotation file path: ", ann_file_path)
                    print("Meta data file path: ", meta_data_file)
                    print("Apnea checkboxes: ", apnea_checkboxes)
                    print("CVD checkboxes: ", cvd_checkboxes)
                    print("AHI: ", AHI)
                    print("Has apnea: ", has_apnea)

                else:
                    pass
                    
    if export:
        meta_data_exported = pd.DataFrame(meta_data_exported)
        meta_data_exported.to_csv("/data/R.Janini_Work/SA-CVD/dataset/meta_data_v3.csv", index=False)
        return meta_data_exported



# def read_meta_file(meta_data_file):
#     edf_file_paths = meta_data_file["edf_file_path"]
#     annotation_file_paths = meta_data_file["annotation_file_path"]
#     meta_data_file_paths = meta_data_file["meta_data_file_path"]
    
#     for i, edf_file_path in enumerate(edf_file_paths):
#         apnea_checkboxes = meta_data_file["apnea_checkboxes"][i].split("; ")
#         print(edf_file_path, annotation_file_paths[i])
#         #check if patient has apnea episodes.
#         AHI = calculate_AHI(edf_file_path, annotation_file_paths[i])
#         print("AHI: ", AHI)
#         break


                 
if __name__ == "__main__":
    # get an argument from the user to extract apnea or hypopnea or both 
    parser = argparse.ArgumentParser()
    parser.add_argument("--apnea", help="Extract apnea events", action="store_true")
    parser.add_argument("--hypopnea", help="Extract hypopnea events", action="store_true")
    parser.add_argument("--mix", help="Extract both apnea and hypopnea events", action="store_true")    

    args = parser.parse_args()
    LOOKFOR = []
    if args.apnea: LOOKFOR = APNEA_EVENTS_NAMES
    if args.hypopnea: LOOKFOR = HYPOPNEA_EVENTS_NAMES
    if args.mix: LOOKFOR = APNEA_EVENTS_NAMES + HYPOPNEA_EVENTS_NAMES

    assert len(LOOKFOR) > 0, "Please specify at least one event type to extract"
    print(f"Extracting {LOOKFOR} events...")

    meta_data_file = export_meta_data(LOOKFOR)  
    # read_meta_file(meta_data_file)