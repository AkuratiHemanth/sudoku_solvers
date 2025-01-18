import streamlit as st
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import matplotlib.pyplot as plt
from grey_binary import colr_to_grey
from image_num import roi_to_num
from st_img_pastebutton import paste
from streamlit_paste_button import paste_image_button
from io import BytesIO
import base64
import numpy as np
import cv2
import io
from PIL import Image


st.title("Sudoku Solver")

#loading the pretrained model
model = load_model("models/digit-49-0.16.keras")

def paste_image_display(image_data):
    # Decode the base64 string
    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)
    bytes_data = BytesIO(binary_data)
    st.image(bytes_data, use_container_width=True)    
    return bytes_data

def show_output(puzzle, sudoku_board):
  # Get the predicted puzzle from the input
  predicted_puzzle = np.argmax(model.predict(puzzle), axis=-1).reshape((9, 9))
  print(predicted_puzzle)
  sudoku = Sudoku(3, 3, board=predicted_puzzle.tolist()) 
  raw_solution = sudoku.solve() #solving the sudoku 
  solution = np.array(raw_solution.board)

  for i in range(9):
    for j in range(9):                  
      pos=(37+114*j, 77+114*i)
      # Check if the predicted cell value is 0
      color = (255, 0, 0)
      if predicted_puzzle[i,j] == 0:
        cv2.putText(sudoku_board,str(solution[i,j]),pos, cv2.FONT_HERSHEY_SIMPLEX, 2, color,5)

  if raw_solution.get_difficulty()>=0:
    st.image(sudoku_board)
  else:
    st.text("Image is unclear or Puuzzle is unsolvable")  


placeholder = st.empty()






upload_image = st.file_uploader(label="Upload image from device",type=['png','jpg'])
paste_image =  paste(label="Paste an image")




pste_btn = False
upld_btn = False
Sudoku_img  = None

col1, col2 = st.columns(2)
with col1:
    st.header("Input")
    if paste_image is not None:
        Sudoku_img = paste_image_display(image_data=paste_image)
        file_bytes = np.asarray(bytearray(Sudoku_img.read()), dtype=np.uint8)
        print(file_bytes)
        Sudoku_img = file_bytes
        pste_btn = st.button("Solve")
    elif upload_image is not None:
        st.image(upload_image)
        file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
        print(file_bytes)
        Sudoku_img = file_bytes
        upld_btn = st.button("Solve")
    else:
        st.write("No image uploaded yet.")

with col2:
    st.header("Output")
    if pste_btn or upld_btn:
            gry,bin = colr_to_grey(Sudoku_img)
            sudoku_num = roi_to_num(gry,bin)
            show_output(sudoku_num,gry)
            if st.button("Clear Content"):
               placeholder.empty()
            


