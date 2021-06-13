import streamlit as st
from streamlit.hashing import _CodeHasher
from SessionState import _SessionState

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server


def main():
    state = _get_state()
    pages = {
        "Dashboard": page_dashboard,
        "Settings": page_settings,
    }

    st.sidebar.title(":floppy_disk: Page states")
    page = st.sidebar.radio("Select your page", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def page_dashboard(state):
    st.title(":chart_with_upwards_trend: Dashboard page")
    display_state_values(state)


def page_settings(state):
    st.title(":wrench: Settings")
    display_state_values(state)

    st.write("---")
    options = ["Hello", "World", "Goodbye"]
    state.input = st.text_input("Set input value.", state.input or "")
    state.slider = st.slider("Set slider value.", 1, 10, state.slider)
    state.radio = st.radio("Set radio value.", options, options.index(state.radio) if state.radio else 0)
    state.checkbox = st.checkbox("Set checkbox value.", state.checkbox)
    state.selectbox = st.selectbox("Select value.", options, options.index(state.selectbox) if state.selectbox else 0)
    state.multiselect = st.multiselect("Select value(s).", options, state.multiselect)

    # Dynamic state assignments
    for i in range(3):
        key = f"State value {i}"
        state[key] = st.slider(f"Set value {i}", 1, 10, state[key])


def display_state_values(state):
    st.write("Input state:", state.input)
    st.write("Slider state:", state.slider)
    st.write("Radio state:", state.radio)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)
    
    for i in range(3):
        st.write(f"Value {i}:", state[f"State value {i}"])

    if st.button("Clear state"):
        state.clear()




def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()