import os


def streamlit_cmd():
    # Change directory to the location of 2_📑_Reference.py
    script_dir = os.path.dirname(os.path.realpath("pages/1_👋_Home_Page.py"))
    os.chdir(script_dir)

    # Run the Streamlit app using streamlit run 2_📑_Reference.py with --no-reload option
    os.system("streamlit run 1_👋_Home_Page.py")

if __name__ == "__main__":
    streamlit_cmd()