# -*- coding: utf-8 -*-
"""
QC analysis.
Step 0 QC: find weird naming.
Step 1 QC: ?
Step 2 QC: obtain video for weird naming assignation
(example: T1 = T2.2)
@author: Tiziana Tocci
"""

import streamlit as st
import os 

try:
    from ..classes.classes_times_qc import FilePreprocesser, QualityController
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from visiowell_pipeline.classes.classes_times_qc import FilePreprocesser, QualityController

st.markdown("""
    <style>
    /* 1. Labels above text inputs */
    div[class*="stTextInput"] label p {
        font-size: 26px !important;
        font-weight: normal !important;
    }
    
    /* 2. Text typed inside text input boxes */
    div[data-baseweb="input"] input {
        font-size: 22px !important;
        font-weight: normal !important;
    }

    /* 3. Current selected value (the number changing next to the label / on hover) */
    div[data-testid="stSlider"] [data-testid="stWidgetLabel"] div,
    div[data-testid="stSlider"] [data-baseweb="slider"] div {
        font-size: 22px !important;
        font-weight: bold !important;
    }


    </style>
""", unsafe_allow_html=True)

def main():
    # QC step0
    st.title("QC for identification of microwells")
    folder_mw = st.text_input("Folder to analyse with microwell results:", ".", key="text1")
    run_find_naming = st.button("Run", key="button1")
    # plot
    name_pos = st.text_input("Insert name of position of which to open the .png file: ", ".")
    run_plot = st.button("Run", key="button2")
    
    if "QC" not in st.session_state or st.session_state.QC is None:
        st.session_state.QC = QualityController()
    QC = st.session_state.QC
    
    if "name_files" not in st.session_state:
        st.session_state.name_files = []
    if "weird_naming_positions" not in st.session_state:
        st.session_state.weird_naming_positions = []
    if "img" not in st.session_state:
        st.session_state.img = None
        
        
    if run_find_naming and folder_mw:
        FP = FilePreprocesser(folder_mw)
        name_files = FP.find_name_files(ext=".txt")
        QC.set_name_files(name_files)
        weird_naming_positions = QC.retrieve_positions_weird_naming()
        
        st.session_state.name_files = name_files
        st.session_state.weird_naming_positions = weird_naming_positions
        
    if st.session_state.name_files:
        st.write(f"Found {len(st.session_state.name_files)} files")
        st.write(f"Found {len(st.session_state.weird_naming_positions)} weird naming positions:")
        st.write(" ".join(st.session_state.weird_naming_positions))
    
    if run_plot and name_pos and QC:
        img = QC.load_fig_naming(folder_mw, name_pos)
        st.session_state.img = img
    if not st.session_state.img is None:
        st.image(st.session_state.img , caption=f"Loade image for position: {name_pos}",\
                     use_container_width=True)
        
    # QC step1
    st.title("QC for identification of cells")
    folder_mw0 = st.text_input("Folder to analyse with microwell results:", ".", key="text4")
    folder_r0 = st.text_input("Folder to analyse with cell detection results:", ".", key="text5")
    pos = st.text_input("Position to analyze:", ".")
    mw = st.text_input("Microwell to analyze:", ".")
    run_bb = st.button("Run", key="button0")
    
    if "video" not in st.session_state:
        st.session_state.video = None
        
    if run_bb and QC:
        video = QC.visualize_bb_score(pos, mw, folder_mw0, folder_r0, (320,320))
        st.session_state.video = video
        
    if st.session_state.video is not None:
        frame_index = st.slider('Choose frame', 0, st.session_state.video.shape[-1]-1)
        st.image(st.session_state.video[:,:,:,frame_index])
        
    
    # QC step2
    st.title("QC for times of division retrieval")
    full_path_output = st.text_input("Insert full path to the output file: ", ".")
    imaging_frequency = st.number_input("Insert imaging frequency:", value=1,\
                                        min_value=1, max_value=10)
    run_times = st.button("Run", key="button3")
    
    if "df_weird_times" not in st.session_state:
        st.session_state.df_weird_times = None
        
    if run_times and imaging_frequency and QC:
        FP1 = FilePreprocesser(full_path_output)
        df = FP1.load_csv()
        QC.set_df_original(df)
        df_weird_times = QC.retrieve_weird_times(int(imaging_frequency))
        st.session_state.df_weird_times = df_weird_times
    
    if "frames" not in st.session_state:
        st.session_state.frames = None
    if "fig" not in st.session_state:
        st.session_state.fig = None
        
    if st.session_state.df_weird_times is not None:
        st.write("Microwells for which the times of division are suspicious")
        st.write(st.session_state.df_weird_times)
        st.write("Number of microwells for which T1=T2.1:")
        st.write((st.session_state.df_weird_times['Type'] == 'T1').sum())
        st.write("Number of microwells for which T2.2=T3.1:")
        st.write((st.session_state.df_weird_times['Type'] == 'T2.2').sum())
        if len(st.session_state.df_weird_times) > 0:
            # visualize a video
            num_row = st.number_input("Insert row df:", value=0,\
                min_value=0, max_value=len(st.session_state.df_weird_times)-1)
            folder_mw1 = st.text_input("Folder to analyse with microwell results:", ".", key="text2")
            folder_r = st.text_input("Folder to analyse with cell detection results (for video):", ".", key="text3")
            run_video = st.button("Load video", key="button4")
            if run_video:
                frames = QC.visualize_video_weird_times(num_row, folder_mw1, folder_r, 3)
                st.session_state.frames = frames
            # look at curve
            folder_r1 = st.text_input("Folder to analyse with cell detection results (for graph):", ".", key="text6")
            run_curve = st.button("Load curve", key="button5")
            if run_curve:
                fig = QC.load_curve(num_row, folder_r1)
                st.session_state.fig = fig
            
    if st.session_state.frames is not None:
        frame_index = st.slider('Choose frame', 0, st.session_state.frames.shape[-1]-1)
        st.image(st.session_state.frames[:,:,:,frame_index])
        
    if st.session_state.fig is not None:
        st.plotly_chart(st.session_state.fig)
            

if __name__ == "__main__":
    main()