import pandas as pd
import argparse
import pytesseract
import os
import cv2

def read_image(impath, lower_thresh=127, higher_thresh=255, add_OTSU=False):
    """
    Reads image
    :param impath:
    :param lower_thresh:
    :param higher_thresh:
    :param add_OTSU:
    :return: image array
    """
    img = cv2.imread(impath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if add_OTSU:
        _, img = cv2.threshold(img, lower_thresh, higher_thresh, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, img = cv2.threshold(img, lower_thresh, higher_thresh, cv2.THRESH_BINARY)
    return img

def save_data(data_dict, filename, filepath='data', pickle_protocol=3, save_txt=True):
    """
    Saves pickle dataframe and labelling ready text file
    :param data_dict: data dictionary to saved as dataframe + text file
    :param filename: Filename to be saved with
    :param filepath: Filepath to be saved at, defaults to data folder
    :param pickle_protocol: Pickle protocol, default 3
    :param save_txt: Weather to save text file too, default True
    :return:
    """
    if not filename.lower().endswith('.pkl'):
        filename = f"{filename}.pkl"

    data_df = pd.DataFrame(
        data={"filepath": data_dict.keys(), "ocr_text": data_dict.values()})
    data_df.to_pickle(os.path.join(filepath, filename), protocol=pickle_protocol)
    if save_txt:
        pd.Series(data_dict.values()).to_csv(filepath, f"{os.path.splitext(filename)[0]}.txt",
                                                         index=False, sep=' ', mode='a', header=False)

def ocr(datapath_list:list, allowed_image_file_extns):
    """
    Ocr image with pytesseract
    :param datapath_list:
    :param allowed_image_file_extns:
    :return:
    """
    ocr_dict = dict()
    for filepath in datapath_list:
        if os.path.splitext(filepath)[-1].lower() in allowed_image_file_extns:
            if filepath not in ocr_dict.keys():
                image = read_image(filepath)
                ocr_dict[filepath] = pytesseract.image_to_string(image)
    return ocr_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Input Values")
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default='data/images',
        type=str,
        help="Directory path containing json annotation files",
    )

    parser.add_argument(
        "--allowed-image-ext",
        dest="allowed_image_file_extns",
        default='.jpg, .jpeg, .png',
        type=str,
        help="Allowed image file extensions separted by commas. Default - .jpg, .jpeg, .png",
    )

    parser.add_argument(
        "--save-dir-path",
        dest="save_dir_path",
        default='data',
        type=str,
        help="Directory path to save dataset pickle file",
    )

    parser.add_argument(
        "--filename",
        dest="filename",
        default='ocr_data.pkl',
        type=str,
        help="Name of pickle file to save",
    )

    parser.add_argument(
        "--pickle-protocol",
        dest="pickle_protocol",
        default=3,
        type=int,
        help="Pickle Protocol",
    )

    args = parser.parse_args()

    datafiles_list = os.listdir(args.data_dir)
    datapath_list = [os.path.join(args.data_dir, filename) for filename in datafiles_list]
    ocr_dict = ocr(datapath_list=datapath_list, allowed_image_file_extns=args.allowed_image_file_extns)
    save_data(ocr_dict, args.save_dir_path, args.filename, pickle_protocol=3, save_txt=True)
