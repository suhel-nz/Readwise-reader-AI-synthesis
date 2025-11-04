# newsletter_synthesis_app/pages/3_Prompt_Editor.py

import streamlit as st
from utils import database, prompts

st.set_page_config(page_title="Prompt Editor", layout="wide")

st.title("✍️ Prompt Editor")
st.info("View and modify the instruction templates used by the AI processing pipeline.")

# --- Load Prompts ---
all_prompts = database.get_all_prompts()
prompt_names = [p['name'] for p in all_prompts]

if not all_prompts:
    st.warning("No prompts found in the database. Trying to re-initialize.")
    prompts.initialize_prompts()
    all_prompts = database.get_all_prompts()
    prompt_names = [p['name'] for p in all_prompts]

# --- UI ---
selected_prompt_name = st.selectbox("Select a prompt to view/edit:", options=prompt_names)

if selected_prompt_name:
    # Find the selected prompt data
    selected_prompt = next((p for p in all_prompts if p['name'] == selected_prompt_name), None)
    
    if selected_prompt:
        st.subheader(f"Editing: `{selected_prompt['name']}`")
        st.markdown(f"**Description:** *{selected_prompt['description']}*")
        st.markdown("---")
        
        prompt_template = st.text_area(
            "Prompt Template",
            value=selected_prompt['template'],
            height=400,
            key=f"prompt_text_{selected_prompt['name']}"
        )
        
        if st.button("Save Changes", type="primary"):
            try:
                database.update_prompt(selected_prompt['name'], prompt_template)
                st.success(f"Prompt '{selected_prompt['name']}' updated successfully!")
                # Clear cache for get_prompt_template if you implement it
            except Exception as e:
                st.error(f"Failed to save prompt: {e}")