import pandas as pd
import streamlit as st
import urllib.parse


df = pd.read_csv("cleaned_laptop_data.csv")

st.set_page_config(page_title="💻 VFM Laptop Recommender", layout="wide")

st.title("🔍 Laptop Value for Money (VFM) Recommender")

# Sidebar for budget input
min_price = st.sidebar.number_input("Minimum Budget ($)", value=300)
max_price = st.sidebar.number_input("Maximum Budget ($)", value=1500)
count = st.sidebar.number_input("Number of Laptops to Display", min_value=1, max_value=50, value=10)

# Filter by price range
filtered_df = df[(df['price_dollar'] >= min_price) & (df['price_dollar'] <= max_price)]

if filtered_df.empty:
    st.warning("❌ No laptops found in this budget range.")
else:
    # Sort by VFM score
    top_vfm = filtered_df.sort_values(by="vfm_score", ascending=False).head(count)

    st.subheader(f"🎯 `Top 10` VFM Laptops from `{min_price}` USD to `{max_price}` USD")

    index = 0
    for idx, row in top_vfm.iterrows():
        brand = row['brand']
        model = row['model']
        ram = row['ram']
        cpu = row['cpu']
        gpu = row['graphics']
        storage = row['harddisk']
        screen = row['screen_size']
        os = row['OS']
        rating = row['rating'] if not pd.isna(row['rating']) else "N/A"
        price = row['price']
        features = row['special_features'] if pd.notna(row['special_features']) else "N/A"

        # Construct search query (with commas and pluses)
        search_query = f"{brand} + {model} + {ram} + {cpu} + {gpu} + {storage} + Amazon"
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(search_query)}"

        # Display block
        st.markdown(f"""
        ---
        ### 🔹{index + 1}. {brand} {model}
        - 💵 **Price:** {price}
        -  **CPU:** `{cpu}`
        - 🎮 **Graphics:** `{gpu}`
        - 💾 **RAM:** `{ram}`
        - 💾 **Storage:** `{storage}`
        - 🖥️ **Screen Size:** `{screen}"`  
        -  **Operating System:** `{os}`
        - 🌟 **Rating:** `{rating}`
        - ✨ **Special Features:** `{features}`
        - 📊 **VFM Score:** `{row['vfm_score']:.4f}`
        - # Web Search [click here]({search_url})
        """, unsafe_allow_html=True)

        index += 1
    st.markdown("---")
