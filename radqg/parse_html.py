##########################################################################################
# Description: A script containing the HTML parser functions.
##########################################################################################

import os
import re
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------------------
# retrieve_figures


def retrieve_figures(root_directory: str) -> list[dict]:
    """A function to retrieve figures from a given directory of saved RadioGraphics
    articles in the format of HTML files."""

    # Regular expression to match the specific format of the figures
    image_pattern = re.compile(r"images_medium_rg\.\d+\.fig\d+[a-z]?\.gif")

    # Iterating over each file in the directory
    figures_list = list()
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

                                figures_list.append(
                                    {
                                        "figure_name": figure_name,
                                        "caption_text": caption_text,
                                        "figure_path": image_path,
                                        "article_file_name": entry,
                                    }
                                )

    return figures_list


# ----------------------------------------------------------------------------------------
# retrieve_fulltexts


def retrieve_fulltexts(root_directory: str) -> list[dict]:
    """A function to retrieve full texts from a given directory of saved RadioGraphics
    articles in the format of HTML files."""

    # Traverse the root directory to get all HTML files
    articles_list = list()
    for file in os.listdir(root_directory):
        if file.endswith(".html"):
            file_path = os.path.join(root_directory, file)

            # Open and parse the HTML file
            with open(file_path, "r", encoding="utf-8") as html_file:
                soup = BeautifulSoup(html_file, "html.parser")

                # Extract title
                title_tag = soup.find("h1", class_="citation__title")
                title_text = title_tag.get_text() if title_tag else ""

                # Extract main article content
                article_tag = soup.find("article")
                texts = []
                if article_tag:
                    for p_tag in article_tag.find_all("p"):
                        # Exclude text within figure and figcaption tags
                        if p_tag.find_parent("figure") or p_tag.find_parent(
                            "figcaption"
                        ):
                            continue
                        texts.append(p_tag.get_text())

                # Concatenate, and replace multiple spaces with a single space
                full_text = title_text + " " + " ".join(texts)
                full_text = re.sub(
                    " +", " ", full_text
                )  # Replace multiple spaces with a single space

                # Save the cleaned text in the dictionary
                articles_list.append(
                    {
                        "article_file_path": file_path,
                        "article_file_name": file,
                        "article_full_text": full_text,
                    }
                )

    return articles_list


# ----------------------------------------------------------------------------------------
# _inspect_tags


def _inspect_tags(html_filepath: str):
    """A function to inspect the tags and their attributes in a given HTML file."""

    with open(html_filepath, "r", encoding="utf-8") as html_file:
        soup = BeautifulSoup(html_file, "html.parser")

        # Finding and printing unique tags and their attributes
        tags_info = {}
        for tag in soup.find_all(True):
            tag_name = tag.name
            attrs = tag.attrs

            # Only save the attributes of the first occurrence of each tag
            if tag_name not in tags_info:
                tags_info[tag_name] = attrs

        # Printing the extracted tag information
        for tag_name, attributes in tags_info.items():
            print(f"Tag: {tag_name} Attributes: {attributes}")
