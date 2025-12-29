import os
import cv2
import pdfplumber
import numpy as np

def checkbox_analysis(meta_data_file_path, SPECIFIED_AREA, reader):
    with pdfplumber.open(meta_data_file_path) as pdf:
        rects = []
        num_pages = len(pdf.pages)

        for i in range(num_pages):
            page = pdf.pages[1]
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

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area > 1000:
                    y += 5
                    h -= 10
                    x += 5
                    w -= 10
                    rect = (x, y, w, h)
                    rects.append(rect)

            closest_area = SPECIFIED_AREA
            closest_rect = None

            for rect in rects:
                x, y, w, h = rect
                if (
                    x < cv2_image.shape[1] / 2
                ):  # get the boxes that are on the left side of the page
                    area = w * h
                    if abs(area - SPECIFIED_AREA) < closest_area:
                        closest_area = abs(area - SPECIFIED_AREA)
                        closest_rect = rect

            x, y, w, h = closest_rect

            binary_cropped_image = binary[y : y + h, x : x + w]
            raw_cropped_img = cv2_image[y : y + h, x : x + w]

        _, otsu_binary = cv2.threshold(raw_cropped_img, 240, 255, cv2.THRESH_BINARY_INV)

        blur = cv2.GaussianBlur(otsu_binary, (5, 5), 0)
        edges = cv2.Canny(blur, 240, 255)

        checked_boxes = []
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area > 500 and area < 1000:
                cropped_box = otsu_binary[y : y + h, x : x + w]

                white_pixels = cv2.countNonZero(cropped_box)
                total_pixels = w * h

                if (
                    white_pixels / total_pixels > 0.5
                ):  # if more than 50% white pixels then it is checked
                    # crop the raw cropped image to include the text next to it
                    x_text = x + w
                    w_text = int(raw_cropped_img.shape[1] / 2) - 50

                    # add top and buttom extra 10 pixels
                    y -= 10
                    h += 20
                    text_box = binary_cropped_image[y : y + h, x_text : x_text + w_text]

                    text = reader.readtext(text_box)[0][1].title()  # OCR the text box
                    checked_boxes.append(text)
                else:
                    pass

    return checked_boxes
