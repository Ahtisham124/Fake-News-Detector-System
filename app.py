import hashlib

import pandas as pd
import streamlit as st

from src.detector import load_or_train_model, predict_text
from src.news_scraper import available_source_labels, fetch_live_news


st.set_page_config(
    page_title="Fake news Detection system",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_or_train_model()


@st.cache_data(ttl=300)
def get_latest_news(source_ids: tuple[str, ...], limit_per_feed: int):
    return fetch_live_news(source_ids=source_ids, limit_per_feed=limit_per_feed)


def show_prediction(text: str) -> None:
    result = predict_text(model, text)
    st.metric("Prediction", result.label.title(), f"{result.confidence:.1%} confidence")
    st.progress(result.confidence)
    st.caption(f"Fake: {result.fake_probability:.1%} | Real: {result.real_probability:.1%}")

    if result.label == "FAKE":
        st.error("This looks suspicious. Please verify it with trusted sources.")
    else:
        st.success("This looks like real news based on the current model.")


def classify_news(news_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in news_df.itertuples(index=False):
        result = predict_text(model, item.text)
        rows.append(
            {
                "source": item.source,
                "title": item.title,
                "prediction": result.label,
                "mark": "FAKE" if result.label == "FAKE" else "REAL",
                "confidence": f"{result.confidence:.1%}",
                "fake_probability": f"{result.fake_probability:.1%}",
                "real_probability": f"{result.real_probability:.1%}",
                "published": item.published,
                "link": item.link,
            }
        )
    return pd.DataFrame(rows)


def stable_text_key(text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"live_news_text_{digest}"


st.title("Fake news Detection system")
st.caption("Simple ML app using live RSS headlines from Times of India and Indian Express.")

model, metrics, model_source = get_model()
source_labels = available_source_labels()

with st.sidebar:
    st.header("Live news")
    selected_names = st.multiselect(
        "Sources",
        options=list(source_labels.values()),
        default=list(source_labels.values()),
    )
    selected_source_ids = tuple(
        source_id
        for source_id, source_name in source_labels.items()
        if source_name in selected_names
    )
    limit_per_feed = st.slider("Items per feed", min_value=3, max_value=20, value=8)

    if st.button("Refresh live data", use_container_width=True):
        get_latest_news.clear()
        st.rerun()

    st.divider()
    st.header("Model")
    st.caption(model_source)
    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    st.metric("Training rows", int(metrics.get("total_rows", 0)))


tab_dashboard, tab_live, tab_manual = st.tabs(["Dashboard", "Live News", "Check Text"])

with tab_dashboard:
    if not selected_source_ids:
        st.warning("Select at least one source from the sidebar.")
    else:
        with st.spinner("Fetching and checking latest RSS headlines..."):
            dashboard_news, dashboard_errors = get_latest_news(
                selected_source_ids,
                limit_per_feed,
            )

        if dashboard_errors:
            with st.expander("Feed errors"):
                for error in dashboard_errors:
                    st.write(f"{error.source} / {error.feed}: {error.message}")

        if dashboard_news.empty:
            st.warning("No live news found. Try refreshing again.")
        else:
            dashboard_df = classify_news(dashboard_news)
            fake_count = int((dashboard_df["prediction"] == "FAKE").sum())
            real_count = int((dashboard_df["prediction"] == "REAL").sum())

            col_total, col_fake, col_real = st.columns(3)
            col_total.metric("Checked news", len(dashboard_df))
            col_fake.metric("Fake marked", fake_count)
            col_real.metric("Real marked", real_count)

            st.subheader("Fake and real marks")
            st.dataframe(
                dashboard_df,
                use_container_width=True,
                hide_index=True,
                column_config={"link": st.column_config.LinkColumn("link")},
            )

            fake_df = dashboard_df[dashboard_df["prediction"] == "FAKE"]
            if not fake_df.empty:
                st.subheader("Fake news marked by model")
                st.dataframe(
                    fake_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"link": st.column_config.LinkColumn("link")},
                )
            else:
                st.info("No live headline is currently marked as FAKE by the model.")

with tab_live:
    if not selected_source_ids:
        st.warning("Select at least one source from the sidebar.")
    else:
        with st.spinner("Fetching latest RSS headlines..."):
            live_news, errors = get_latest_news(selected_source_ids, limit_per_feed)

        if errors:
            with st.expander("Feed errors"):
                for error in errors:
                    st.write(f"{error.source} / {error.feed}: {error.message}")

        if live_news.empty:
            st.warning("No live news found. Try refreshing again.")
        else:
            st.metric("Live items", len(live_news))

            display_df = live_news[["source", "feed", "title", "published", "link"]]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={"link": st.column_config.LinkColumn("link")},
            )

            options = [
                f"{row.source}: {row.title}"
                for row in live_news.itertuples(index=False)
            ]
            selected_option = st.selectbox("Choose a headline to check", options)
            selected_index = options.index(selected_option)
            selected_item = live_news.iloc[selected_index]

            st.subheader(selected_item["title"])
            if selected_item["summary"]:
                st.write(selected_item["summary"])
            if selected_item["link"]:
                st.link_button("Open original article", selected_item["link"])

            original_text = str(selected_item["text"])
            edited_text = st.text_area(
                "Selected RSS text",
                value=original_text,
                height=160,
                key=stable_text_key(original_text),
            )

            if st.button("Check selected headline", type="primary", use_container_width=True):
                if edited_text != original_text:
                    st.error("Error: this live news text was changed. Please refresh or use the original RSS text.")
                else:
                    show_prediction(original_text)

with tab_manual:
    sample_text = (
        "Paste any news headline or short article here to check whether it looks "
        "fake or real."
    )
    user_text = st.text_area(
        "News text",
        value=sample_text,
        height=220,
        placeholder="Paste a headline or article summary...",
    )

    if st.button("Check news", type="primary", use_container_width=True):
        if user_text.strip():
            show_prediction(user_text)
        else:
            st.warning("Enter some news text first.")
