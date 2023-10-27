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


def retrieve_fulltexts(root_directory):
    articles_dict = {}

    # Traverse the root directory to get all HTML files
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)

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
                    articles_dict[file_path] = full_text

    return articles_dict


def inspect_tags(html_filepath):
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


# Uncomment the following lines to test the functions
# inspect_tags('/research/projects/m221279_Pouria/RadQG/data/html_articles/Internal Hernias in the Era of Multidetector CT_ Correlation of Imaging and Surgical Findings _ RadioGraphics.html')
# full_text_dict = retrieve_fulltexts('/research/projects/m221279_Pouria/RadQG/data/html_articles')
# print(list(full_text_dict.items())[1][-1])
