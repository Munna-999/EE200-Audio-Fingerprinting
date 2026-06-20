import streamlit as st
import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import librosa
import matplotlib.pyplot as plt
import pickle
import time
import os
from collections import defaultdict

# ==========================================
# 1. PAGE SETUP
# ==========================================
# Set the page to wide mode to match the demo interface
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")

# ==========================================
# 2. CORE FUNCTIONS (From Q3A)
# ==========================================
@st.cache_data
def load_db():
    try:
        with open('song_database.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("🚨 song_database.pkl not found! Please run the indexing script first.")
        return defaultdict(list)

def generate_hashes(peak_times, peak_freqs, fan_out=3):
    hashes = []
    sorted_indices = np.argsort(peak_times)
    times = peak_times[sorted_indices]
    freqs = peak_freqs[sorted_indices]
    
    for i in range(len(times)):
        anchor_time = float(times[i])
        anchor_freq = float(freqs[i])
        for j in range(1, fan_out + 1):
            if i + j < len(times):
                target_time = float(times[i + j])
                target_freq = float(freqs[i + j])
                time_delta = round(target_time - anchor_time, 3)
                if time_delta > 0:
                    hashes.append(((anchor_freq, target_freq, time_delta), anchor_time))
    return hashes

def match_query(query_hashes, database, threshold=5):
    matches = defaultdict(lambda: defaultdict(int))
    
    for q_hash in query_hashes:
        fingerprint, query_time = q_hash
        if fingerprint in database:
            for db_song, db_time in database[fingerprint]:
                offset = round(db_time - query_time, 1)
                matches[db_song][offset] += 1
                
    if not matches:
        return "none", 0, {}, []
        
    song_scores = {}
    for song, offsets in matches.items():
        if len(offsets) > 0:
            song_scores[song] = max(offsets.values())
            
    sorted_candidates = sorted(song_scores.items(), key=lambda item: item[1], reverse=True)
    best_song, best_score = sorted_candidates[0]
    best_histogram = matches[best_song]
    
    if best_score < threshold:
        return "none", best_score, {}, sorted_candidates[:5]
        
    return best_song, best_score, best_histogram, sorted_candidates[:5]

# ==========================================
# 3. UI HEADER & TABS
# ==========================================
music_db = load_db()

st.title("EE200: Audio Fingerprinting")
st.caption("SIGNALS, SYSTEMS & NETWORKS · PROJECT DEMO")
st.markdown("Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")

tab1, tab2, tab3 = st.tabs(["❖ LIBRARY", "◎ IDENTIFY", "▤ BATCH"])

# ==========================================
# TAB 1: VIEW DATABASE (Neon Grid)
# ==========================================
with tab1:
    st.info("Song indexing is managed by the admin.  \nDrop a clip in the Identify tab to test the library.", icon="ℹ️")
    st.markdown("### IN THE DATABASE")
    
    if music_db:
        with st.spinner("Loading library visuals..."):
            song_stats = defaultdict(lambda: {'count': 0, 'points': []})
            
            for fingerprint, matches in music_db.items():
                anchor_freq = fingerprint[0] 
                for db_song, db_time in matches:
                    song_stats[db_song]['count'] += 1
                    if len(song_stats[db_song]['points']) < 300:
                        song_stats[db_song]['points'].append((db_time, anchor_freq))
                        
            sorted_songs = sorted(song_stats.keys())
            cols = st.columns(4)
            neon_colors = ['#00d4b2', '#f39c12', '#9b59b6', '#ff4b4b', '#f1c40f', '#00a8ff']
            
            for i, song in enumerate(sorted_songs):
                with cols[i % 4]:
                    with st.container(border=True):
                        points = np.array(song_stats[song]['points'])
                        if len(points) > 0:
                            fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
                            c = neon_colors[i % len(neon_colors)] 
                            ax.scatter(points[:, 0], points[:, 1], s=1.5, color=c, alpha=0.8)
                            ax.axis('off')
                            fig.patch.set_alpha(0.0)
                            ax.patch.set_alpha(0.0)
                            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)
                            
                        st.markdown(f"**{song[:25]}**" + ("..." if len(song) > 25 else ""))
                        st.caption(f"{song_stats[song]['count']:,} hashes")
    else:
        st.warning("Database is empty.")

# ==========================================
# TAB 2: SINGLE-CLIP MODE (Full Video Re-creation)
# ==========================================
with tab2:
    st.subheader("Identify a clip")
    
    audio_to_process = None
    query_file = st.file_uploader("", type=['mp3', 'wav', 'flac', 'ogg', 'm4a'], key="single_upload")
    if query_file:
        audio_to_process = query_file

    st.markdown("<h6 style='color: gray; margin-top: 20px;'>OR TRY A SAMPLE</h6>", unsafe_allow_html=True)
    
    for i in range(1, 6):
        sample_path = f"sample{i}.wav" 
        col1, col2, col3 = st.columns([1, 4, 1])
        with col1:
            st.write(f"sample{i}")
        with col2:
            if os.path.exists(sample_path):
                st.audio(sample_path)
            else:
                st.caption(f"Audio file {sample_path} not found in folder.")
        with col3:
            if st.button("Try", key=f"try_{i}", type="primary", use_container_width=True):
                if os.path.exists(sample_path):
                    audio_to_process = sample_path
                else:
                    st.error(f"Missing {sample_path}")

    if audio_to_process is not None:
        st.divider() 
        
        with st.spinner("Analyzing frequencies..."):
            import time
            t0 = time.time()
            audio, fs = librosa.load(audio_to_process, sr=None, mono=True)
            t1 = time.time()
            
            # Use smaller nperseg for higher time resolution if needed, but keeping your original math
            f, t, Sxx = signal.spectrogram(audio, fs=fs, nperseg=2048)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            query_duration = t[-1] # Needed for the highlight box later
            t2 = time.time()
            
            threshold_db = np.percentile(Sxx_db, 95)
            data_max = ndimage.maximum_filter(Sxx_db, size=20)
            peak_mask = (Sxx_db == data_max) & (Sxx_db > threshold_db)
            peak_f, peak_t = np.where(peak_mask)
            t3 = time.time()
            
            query_hashes = generate_hashes(t[peak_t], f[peak_f])
            t4 = time.time()
            
            best_song, best_score, best_histogram, top_candidates = match_query(query_hashes, music_db)
            t5 = time.time()

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("SPECTROGRAM", f"{int((t2-t1)*1000)} ms")
        m2.metric("CONSTELLATION", f"{int((t3-t2)*1000)} ms", f"{len(peak_t)} peaks")
        m3.metric("HASHING", f"{int((t4-t3)*1000)} ms", f"{len(query_hashes)} hashes")
        m4.metric("DB LOOKUP", f"{int((t5-t4)*1000)} ms", f"{len(music_db)} tracks")
        
        winning_offset = 0
        if best_histogram:
            winning_offset = max(best_histogram, key=best_histogram.get)
            
        m5.metric("SCORING", f"1 ms", f"offset {winning_offset}s")
        m6.metric("TOTAL TIME", f"{int((t5-t0)*1000)} ms")
        
        if best_song != "none":
            st.success(f"##### MATCH FOUND  \n# {best_song}  \n`Cluster score: {best_score}`")
            
            st.markdown("<h6 style='color: gray; margin-top: 20px;'>CANDIDATE SCORES</h6>", unsafe_allow_html=True)
            for candidate, score in top_candidates:
                st.write(f"**{candidate}** - {score}")
            st.divider()
            
            # Dark theme settings for Matplotlib
            plt.style.use('dark_background')
            grid_color = '#2C3E50'
            
            # --- STEP 1: VISUALS ---
            st.markdown("##### STEP 1 ᐧ FEATURE EXTRACTION")
            st.markdown("### From spectrogram to constellation")
            st.caption(f"The clip was converted into a time-frequency map. Only the **{len(peak_t)} most prominent peaks** were kept.")
            
            col_a, col_b = st.columns(2)
            with col_a:
                fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=100)
                ax1.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
                ax1.set_ylabel('freq bin', color='#bdc3c7')
                ax1.set_xlabel('time (s)', color='#bdc3c7')
                ax1.tick_params(colors='#bdc3c7')
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                fig1.patch.set_alpha(0.0)
                ax1.patch.set_alpha(0.0)
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)
                
            with col_b:
                fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=100)
                ax2.scatter(t[peak_t], f[peak_f], c='#00d4b2', s=5, alpha=0.8)
                ax2.set_ylabel('freq bin', color='#bdc3c7')
                ax2.set_xlabel('time (s)', color='#bdc3c7')
                ax2.grid(True, color=grid_color, linestyle='--', alpha=0.5)
                ax2.tick_params(colors='#bdc3c7')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                fig2.patch.set_alpha(0.0)
                ax2.patch.set_alpha(0.0)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            # --- STEP 2: WHERE IN THE SONG? ---
            st.markdown("##### STEP 2 ᐧ DATABASE SEARCH")
            st.markdown("### Where in the song?")
            st.caption(f"The **{len(query_hashes)} fingerprint hashes** were looked up against every indexed track. Below is the full fingerprint of `{best_song}` reconstructed from the database. The highlighted window is exactly where the query clip sits inside the full song.")
            
            # Reconstruct the full song's constellation from the database
            full_song_points = set()
            for fingerprint, matches in music_db.items():
                anchor_freq = fingerprint[0]
                for db_song, db_time in matches:
                    if db_song == best_song:
                        full_song_points.add((db_time, anchor_freq))
            
            if full_song_points:
                fs_points = np.array(list(full_song_points))
                fig_db, ax_db = plt.subplots(figsize=(12, 4), dpi=100)
                # Plot full song dots in cyan
                ax_db.scatter(fs_points[:, 0], fs_points[:, 1], c='#00d4b2', s=3, alpha=0.4)
                
                # Draw the highlight window where the clip matches!
                ax_db.axvspan(winning_offset, winning_offset + query_duration, color='#f1c40f', alpha=0.2)
                ax_db.text(winning_offset, ax_db.get_ylim()[1], 'query clip aligned here', color='#f1c40f', ha='left', va='bottom')
                
                ax_db.set_ylabel('freq bin', color='#bdc3c7', fontsize=12)
                ax_db.set_xlabel('time (s)', color='#bdc3c7', fontsize=12)
                ax_db.grid(True, color=grid_color, linestyle='-', alpha=0.4)
                ax_db.tick_params(colors='#bdc3c7')
                ax_db.spines['top'].set_visible(False)
                ax_db.spines['right'].set_visible(False)
                fig_db.patch.set_alpha(0.0)
                ax_db.patch.set_alpha(0.0)
                st.pyplot(fig_db, use_container_width=True)
                plt.close(fig_db)

            # --- STEP 3: THE PROOF ---
            st.markdown("##### STEP 3 ᐧ THE PROOF")
            st.markdown("### The alignment spike")
            st.caption(f"Every matched hash votes for a time offset. Chance matches scatter randomly, forming a flat noise floor. A genuine match makes them converge: **{best_score} hashes agreed on a single offset**. That spike cannot be a coincidence.")
            
            fig3, ax3 = plt.subplots(figsize=(12, 4), dpi=100)
            offsets = list(best_histogram.keys())
            counts = list(best_histogram.values())
            
            # Plot the "noise floor" in a muted color
            ax3.bar(offsets, counts, width=0.2, color='#e67e22', alpha=0.4)
            
            # Highlight the winning spike in bright orange/yellow
            ax3.bar([winning_offset], [best_score], width=0.3, color='#f39c12', alpha=1.0)
            
            # Add the text annotation pointing to the spike
            ax3.annotate(f"{best_score} hashes\nalign here", 
                         xy=(winning_offset, best_score), 
                         xytext=(winning_offset + 5, best_score * 0.8),
                         color='#f39c12',
                         arrowprops=dict(facecolor='#f39c12', edgecolor='#f39c12', arrowstyle='->'))

            ax3.set_ylabel('# hashes', color='#bdc3c7')
            ax3.set_xlabel('time offset (database time - query time)', color='#bdc3c7')
            ax3.grid(True, color=grid_color, linestyle='-', alpha=0.3)
            ax3.tick_params(colors='#bdc3c7')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            fig3.patch.set_alpha(0.0)
            ax3.patch.set_alpha(0.0)
            
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
            
            # Reset style so it doesn't mess up other tabs
            plt.style.use('default')
            
        else:
            st.error("❌ No confident match found. The recording might be too short or too noisy.")

# ==========================================
# TAB 3: BATCH MODE
# ==========================================
with tab3:
    st.subheader("Identify many clips at once")
    st.markdown("Upload a set of query clips. Each is identified against the **currently indexed library**, and the results are written to a standardised `results.csv` with columns `filename`, `prediction`. The `prediction` is the matched track's filename without its extension, or `none` when no candidate clears the confidence threshold.")
    
    uploaded_files = st.file_uploader("", type=['mp3', 'wav'], accept_multiple_files=True, key="batch_upload")
    
    if uploaded_files and st.button("Run batch", type="primary"):
        st.markdown("### RESULTS")
        results_display = []
        results_csv = []
        
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            audio, fs = librosa.load(file, sr=None, mono=True)
            f, t, Sxx = signal.spectrogram(audio, fs=fs, nperseg=2048)
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            threshold_db = np.percentile(Sxx_db, 95)
            data_max = ndimage.maximum_filter(Sxx_db, size=20)
            peak_mask = (Sxx_db == data_max) & (Sxx_db > threshold_db)
            peak_f, peak_t = np.where(peak_mask)
            
            query_hashes = generate_hashes(t[peak_t], f[peak_f])
            
            # Unpack the 4 returned values from match_query
            best_song, _, _, _ = match_query(query_hashes, music_db)
            
            results_display.append({"FILE": file.name, "PREDICTION": best_song})
            results_csv.append({"filename": file.name, "prediction": best_song})
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        st.table(pd.DataFrame(results_display))
        
        csv_df = pd.DataFrame(results_csv)
        csv = csv_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name='results.csv',
            mime='text/csv',
            type="primary"
        )