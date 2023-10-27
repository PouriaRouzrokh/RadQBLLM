import os
import re

from bs4 import BeautifulSoup


def retrieve_figures(root_directory):
    figures_dict = {}

    # Regular expression to match the specific format of the figures
    image_pattern = re.compile(r"images_medium_rg\.\d+\.fig\d+[a-z]?\.gif")

    # Iterating over each file in the directory
    for entry in os.listdir(root_directory):
        if entry.endswith(".html"):
            html_file_path = os.path.join(root_directory, entry)
            folder_path = os.path.join(root_directory, entry.replace(".html", "_files"))

            with open(html_file_path, "r") as file:
                soup = BeautifulSoup(file, "html.parser")

                # Finding figures and captions
                for figure_tag in soup.find_all("figure"):
                    # Assuming the image is within an <img> tag
                    img_tag = figure_tag.find("img")
                    if img_tag and image_pattern.search(img_tag["src"]):
                        figcaption_tag = figure_tag.find("figcaption")
                        if figcaption_tag:
                            caption_text = " ".join(
                                figcaption_tag.get_text(strip=True).split()
                            )

                            # Extract figure name from the caption text
                            match = re.search(r"(Figure \d+[a-z]?)", caption_text)
                            if match:
                                figure_name = match.group(1)

                                # Creating the absolute image path
                                image_filename = os.path.basename(img_tag["src"])
                                image_path = os.path.abspath(
                                    os.path.join(folder_path, image_filename)
                                )

                                figures_dict[image_path] = [
                                    entry,
                                    figure_name,
                                    caption_text,
                                ]

    return figures_dict
