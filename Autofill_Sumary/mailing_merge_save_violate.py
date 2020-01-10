<<<<<<< HEAD
import os
import sys
from datetime import date
from mailmerge import MailMerge
import xlrd
import pandas as pd
# sys.path.insert(0, '/home/truongdongdo/Desktop/python_function')
csv_sumary_path = "sumary.csv"

def save_case_to_csv():
    csv_sumary_name = 'sumary_11.csv'
    sumary_data = [['STT', 'License', 'Time','Location']]
    sumary_data_row = []
    sumary_data = [['Case_name', 'Amount', 'Img_shape']]
    sumary_data_row.append(case_name)
    sumary_data_row.append(len(img_list_per_case))
    sumary_data_row.append(first_img.shape)
    sumary_data.append(sumary_data_row)
    sumary_data_row = []

    sumary_path = data_path + 'CSV_File/' + csv_sumary_name
    with open(sumary_path, 'w') as SumaryFile:
        writer = csv.writer(SumaryFile)
        writer.writerows(sumary_data)

    SumaryFile.close()

    print(" Sumary writing complete!")



def mailing_merge_save_violate():
    current_row = 0
    sheet_num = 0
    # data_path = "C://Users/ACER/Desktop/Autofill/"
    data_path = "/home/truongdongdo/Desktop/Autofill/"
    # arg 1: Template file path
    docx_path = data_path + "Template.docx"
    # arg 2: Excel file path
    excel_path = data_path + "TemplateValues.xlsx"
    # arg 3: Output file
    output_path = data_path + "Bien_ban/"
    # arg 4: Excel workbook sheet number

    if(os.path.isfile(docx_path)== False or os.path.isfile(excel_path)== False ):
        print("Cannot find input file.")
        sys.exit()
    if(os.path.isfile(output_path) == True):
        print("Output file already exists.")
        sys.exit()

    print("Reading .docx file.")
    document = MailMerge(docx_path)
    print(document.get_merge_fields())

    # Path to the file you want to extract data from
    print("Reading .xlsx file.")
    book = xlrd.open_workbook(excel_path)

    if ((book.nsheets >=sheet_num+1) == False):
        print("Unable to find the sheet number provided.")
        sys.exit()

    # Select the sheet that the data resids in
    work_sheet = book.sheet_by_index(sheet_num)
    finalList = []
    headers = []

    #get the total number of the rows
    num_rows = work_sheet.nrows

    # Format required for mail merge is:
    # List [
    # {Dictrionaty},
    # {Dictrionaty},
    # ....
    # ]

    print("Preparing to merge.")
    while current_row < num_rows:
        dictVal = dict()
        if(current_row == 0):
            for col in range(work_sheet.ncols):
                headers.append(work_sheet.cell_value(current_row,col))
                print(headers)
        else:
            for col in range(work_sheet.ncols):
                dictVal.update({headers[col]:work_sheet.cell_value(current_row,col)})
                print(dictVal)
        if(current_row != 0):
            finalList.append(dictVal)

        current_row+=1

    print(finalList)
    print("Merge operation started.")
    document.merge_pages(finalList)
    print("Saving output file.")
    output_docx_name = "datetime"
    document.write(output_path + output_docx_name + ".docx")

    print("Operation complete successfully.")

if __name__ == '__main__':
=======
import os
import sys
from datetime import date
from mailmerge import MailMerge
import xlrd
import pandas as pd

from time import strftime
from datetime import datetime

# sys.path.insert(0, '/home/truongdongdo/Desktop/python_function')
# csv_sumary_path = "sumary.csv"

# def save_case_to_csv():
#     csv_sumary_name = 'sumary_11.csv'
#     sumary_data = [['STT', 'License', 'Time','Location']]
#     sumary_data_row = []
#     sumary_data = [['Case_name', 'Amount', 'Img_shape']]
#     sumary_data_row.append(case_name)
#     sumary_data_row.append(len(img_list_per_case))
#     sumary_data_row.append(first_img.shape)
#     sumary_data.append(sumary_data_row)
#     sumary_data_row = []

#     sumary_path = data_path + 'CSV_File/' + csv_sumary_name
#     with open(sumary_path, 'w') as SumaryFile:
#         writer = csv.writer(SumaryFile)
#         writer.writerows(sumary_data)

#     SumaryFile.close()

#     print(" Sumary writing complete!")



def mailing_merge_save_violate(work_sheet, index_row ,document, output_path):
    
    if(os.path.isfile(output_path) == True):
        print("Output file already exists.")
        sys.exit()

    # Select the sheet that the data resids in
    finalList = []
    headers = []

    #get the total number of the rows
    # num_rows = work_sheet.nrows

    # Format required for mail merge is:
    # List [
    # {Dictrionaty},
    # {Dictrionaty},
    # ....
    # ]

    current_row = 0
    print("Preparing to merge.")
    # while current_row < num_rows:
    dictVal = dict()

    # Make header
    # if(current_row == 0):
    header_row = 0
    for col in range(work_sheet.ncols):
        headers.append(work_sheet.cell_value(header_row,col))
        print(headers)

    # Update dictVal
    for col in range(work_sheet.ncols):
        dictVal.update({headers[col]:work_sheet.cell_value(index_row,col)})
        print(dictVal)

    print("Merge operation started.")
    document.merge_pages(dictVal)
    print("Saving output file.")

    time = datetime.now().strftime('%d/%m/%Y %H:%M')
    bien_so = work_sheet.cell_value(index_row,0) + work_sheet.cell_value(index_row,1) 
    output_docx_name = bien_so + "_" + time
    document.write(output_path + output_docx_name + ".docx")

    print("Operation complete successfully.")

if __name__ == '__main__':

        # data_path = "C://Users/ACER/Desktop/Autofill/"
    data_path = "/home/truongdongdo/Documents/GitKraken/LicensePlateRecognition/Autofill_Sumary/"
    # arg 1: Template file path
    docx_path = data_path + "Bien_ban_vi_pham.docx"
    # arg 2: Excel file path
    excel_path = data_path + "Information_License_plate.xlsx"
    # arg 3: Output file
    output_path = data_path + "Bien_ban/"
    # arg 4: Excel workbook sheet number

    # Doc kiem tra duong dan
    if(os.path.isfile(docx_path)== False or os.path.isfile(excel_path)== False ):
        print("Cannot find input file.")
        sys.exit()

    # Doc file Word
    print("Reading .docx file.")
    document = MailMerge(docx_path)
    print(document.get_merge_fields())

    # Doc file excel
    print("Reading .xlsx file.")
    book = xlrd.open_workbook(excel_path)

    sheet_num = 0
    if ((book.nsheets >=sheet_num+1) == False):
        print("Unable to find the sheet number provided.")
    sys.exit()

    work_sheet = book.sheet_by_index(sheet_num)

    
    index_row = 2
    mailing_merge_save_violate(work_sheet, index_row ,document, output_path)





>>>>>>> c597cb084bdf75d3f74a70fb3c961df3262538b9
    mailing_merge_save_violate()