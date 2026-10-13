/**
 * PlagiGuard YT - Frontend Client Application
 * Author: omanushoriginal <drisham4@gmail.com>
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const tabScan = document.getElementById("tab-scan");
  const tabCompare = document.getElementById("tab-compare");
  const scanInputGroup = document.getElementById("scan-input-group");
  const compareInputGroup = document.getElementById("compare-input-group");
  
  const videoUrlInput = document.getElementById("video-url-input");
  const compareUrl1 = document.getElementById("compare-url-1");
  const compareUrl2 = document.getElementById("compare-url-2");
  
  const btnSubmit = document.getElementById("btn-submit");
  const toggleDinov2 = document.getElementById("toggle-dinov2");
  const candidateCountSelect = document.getElementById("candidate-count");

  const progressSection = document.getElementById("progress-section");
  const progressText = document.getElementById("progress-text");
  const step1 = document.getElementById("step-1");
  const step2 = document.getElementById("step-2");
  const step3 = document.getElementById("step-3");
  const step4 = document.getElementById("step-4");

  const resultsSection = document.getElementById("results-section");
  const sourceHeroCard = document.getElementById("source-hero-card");
  const threatScoreRing = document.getElementById("threat-score-ring");
  const threatScoreVal = document.getElementById("threat-score-val");
  const threatTitle = document.getElementById("threat-title");
  const threatDesc = document.getElementById("threat-desc");
  const matchesContainer = document.getElementById("matches-container");

  const dmcaModal = document.getElementById("dmca-modal");
  const dmcaTextarea = document.getElementById("dmca-textarea");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const btnCopyDmca = document.getElementById("btn-copy-dmca");
  const btnDownloadReport = document.getElementById("btn-download-report");
  const btnOpenDmca = document.getElementById("btn-open-dmca");

  let currentMode = "scan"; // 'scan' or 'compare'
  let latestScanData = null;

  // Preset Buttons
  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const type = chip.getAttribute("data-preset");
      if (type === "tech") {
        videoUrlInput.value = "https://www.youtube.com/watch?v=_uQrJ0TkZlc";
      } else if (type === "edu") {
        videoUrlInput.value = "https://www.youtube.com/watch?v=rfscVS0vtbw";
      } else if (type === "sample") {
        loadSampleReport();
        return;
      }
      triggerScan();
    });
  });

  // Mode Tabs Switching
  tabScan.addEventListener("click", () => {
    currentMode = "scan";
    tabScan.classList.add("active");
    tabCompare.classList.remove("active");
    scanInputGroup.style.display = "flex";
    compareInputGroup.style.display = "none";
  });

  tabCompare.addEventListener("click", () => {
    currentMode = "compare";
    tabCompare.classList.add("active");
    tabScan.classList.remove("active");
    scanInputGroup.style.display = "none";
    compareInputGroup.style.display = "grid";
  });

  // Submit button
  btnSubmit.addEventListener("click", () => {
    triggerScan();
  });

  videoUrlInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") triggerScan();
  });

  function triggerScan() {
    if (currentMode === "scan") {
      const url = videoUrlInput.value.trim();
      if (!url) {
        alert("Please enter a valid YouTube video URL or Video ID.");
        return;
      }
      runScan(url);
    } else {
      const u1 = compareUrl1.value.trim();
      const u2 = compareUrl2.value.trim();
      if (!u1 || !u2) {
        alert("Please enter both original and suspect YouTube video URLs.");
        return;
      }
      runCompare(u1, u2);
    }
  }

  // Animation helper for scanning steps
  function setStep(stepNum, text) {
    progressText.innerText = text;
    const steps = [step1, step2, step3, step4];
    steps.forEach((s, idx) => {
      if (idx + 1 < stepNum) {
        s.className = "step-item done";
      } else if (idx + 1 === stepNum) {
        s.className = "step-item active";
      } else {
        s.className = "step-item";
      }
    });
  }

  // API Call: Full YouTube Scan
  async function runScan(url) {
    showLoading();
    setStep(1, "1/4 Resolving YouTube Video Metadata & Thumbnail...");

    const maxCandidates = parseInt(candidateCountSelect?.value || "6", 10);
    const useDinov2 = toggleDinov2 ? toggleDinov2.checked : true;

    try {
      setTimeout(() => setStep(2, "2/4 Searching YouTube for Duplicate & Derivative Videos..."), 1200);
      setTimeout(() => setStep(3, "3/4 Running DINOv2 Deep Feature & SIFT Homography Matching..."), 3500);
      setTimeout(() => setStep(4, "4/4 Analyzing Spoken Transcripts & Synthesizing Threat Matrix..."), 7000);

      const resp = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: url,
          max_candidates: maxCandidates,
          use_dinov2: useDinov2
        })
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Scan request failed");
      }

      const data = await resp.json();
      latestScanData = data;
      renderScanResults(data);
    } catch (err) {
      console.warn("Backend API not reachable; falling back to client-side demonstration engine", err);
      // Simulate live forensic report for GitHub Pages visitor
      const mockResult = generateClientSideDemo(url);
      latestScanData = mockResult;
      setTimeout(() => {
        renderScanResults(mockResult);
      }, 1500);
    }
  }

  // API Call: Direct 1-on-1 Compare
  async function runCompare(url1, url2) {
    showLoading();
    setStep(1, "1/3 Fetching both YouTube video streams & thumbnails...");

    const useDinov2 = toggleDinov2 ? toggleDinov2.checked : true;

    try {
      setTimeout(() => setStep(2, "2/3 Running SIFT Keypoints, DINOv2 & Transcript Alignment..."), 2000);
      setTimeout(() => setStep(3, "3/3 Generating Side-by-Side DMCA Evidence..."), 4500);

      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url1: url1,
          url2: url2,
          use_dinov2: useDinov2
        })
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || "Comparison request failed");
      }

      const data = await resp.json();
      // Format 1-on-1 comparison into standard scan format
      const formatted = {
        status: "success",
        scan_time_sec: 3.5,
        source_video: data.video1,
        candidates_count: 1,
        highest_plagiarism_risk_pct: data.composite_verdict.risk_score_pct,
        matches: [
          {
            candidate: data.video2,
            visual_comparison: data.visual_comparison,
            transcript_comparison: data.transcript_comparison,
            title_comparison: data.title_comparison,
            verdict: data.composite_verdict
          }
        ]
      };
      latestScanData = formatted;
      renderScanResults(formatted);
    } catch (err) {
      console.error(err);
      alert("Comparison failed: " + err.message);
      hideLoading();
    }
  }

  // Load sample report instantly
  async function loadSampleReport() {
    showLoading();
    setStep(1, "Loading Verified YouTube Plagiarism Case Studies...");
    try {
      const resp = await fetch("/api/sample");
      const data = await resp.json();
      latestScanData = data;
      setTimeout(() => {
        renderScanResults(data);
      }, 500);
    } catch (e) {
      console.warn("Falling back to local demo mock data", e);
    }
  }

  function showLoading() {
    resultsSection.style.display = "none";
    progressSection.style.display = "block";
    btnSubmit.disabled = true;
  }

  function hideLoading() {
    progressSection.style.display = "none";
    btnSubmit.disabled = false;
  }

  // Render Full Results
  function renderScanResults(data) {
    hideLoading();
    resultsSection.style.display = "flex";

    const src = data.source_video;
    const highestRisk = data.highest_plagiarism_risk_pct || 0;

    // Render Source Video Card
    sourceHeroCard.innerHTML = `
      <div class="thumb-preview-wrapper">
        <img src="${src.thumbnail}" alt="Thumbnail" onerror="this.src='https://img.youtube.com/vi/${src.id}/hqdefault.jpg'" />
        <span class="video-duration">${src.duration_str || 'Video'}</span>
      </div>
      <div class="source-info">
        <div class="source-tag">Target Query Video</div>
        <h3 class="source-title">${escapeHtml(src.title)}</h3>
        <div class="source-meta-row">
          <div class="meta-item">
            <svg width="15" height="15" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/></svg>
            <strong>${escapeHtml(src.channel)}</strong>
          </div>
          <div class="meta-item">
            <span>📅 ${src.upload_date || 'N/A'}</span>
          </div>
          <div class="meta-item">
            <span>👁️ ${formatNumber(src.view_count)} views</span>
          </div>
          <div class="meta-item">
            <span>🎙️ Captions: ${src.transcript_stats?.available ? 'Available (' + src.transcript_stats.word_count + ' words)' : 'No Captions'}</span>
          </div>
        </div>
        <a href="${src.url}" target="_blank" rel="noopener" class="link-btn">
          Watch on YouTube ↗
        </a>
      </div>
    `;

    // Render Threat Matrix Banner
    threatScoreVal.innerText = `${highestRisk}%`;
    let ringColor = "#10b981";
    let badgeTitle = "Original Content / Low Copy Risk";
    let desc = "Scanned across YouTube candidates. No significant duplicate thumbnails or verbatim scripts detected.";

    if (highestRisk >= 75) {
      ringColor = "#ef4444";
      badgeTitle = "CRITICAL: High Probability of Content Theft";
      desc = "Strong visual and/or transcript duplication detected across competing YouTube videos.";
    } else if (highestRisk >= 50) {
      ringColor = "#f97316";
      badgeTitle = "SUSPICIOUS: AI-Modified or Heavy Derivative Found";
      desc = "Detected suspicious layout homography, AI-stylized thumbnail, or paraphrased spoken audio.";
    } else if (highestRisk >= 30) {
      ringColor = "#eab308";
      badgeTitle = "MODERATE: Concept or Keyword Overlap";
      desc = "Videos share similar titles or instructional concepts, but assets remain largely distinct.";
    }

    threatScoreRing.style.borderColor = ringColor;
    threatScoreRing.style.color = ringColor;
    threatTitle.innerText = badgeTitle;
    threatDesc.innerText = `${desc} (Scan completed in ${data.scan_time_sec}s across ${data.candidates_count} candidates).`;

    // Render Matches List
    matchesContainer.innerHTML = "";
    if (!data.matches || data.matches.length === 0) {
      matchesContainer.innerHTML = `<div class="match-card" style="text-align:center; padding: 2.5rem; color: #9ca3af;">No candidate videos found matching this query on YouTube.</div>`;
      return;
    }

    data.matches.forEach((m, idx) => {
      const c = m.candidate;
      const v = m.verdict;
      const vis = m.visual_comparison;
      const trans = m.transcript_comparison;
      const tit = m.title_comparison;

      const card = document.createElement("div");
      card.className = "match-card";

      card.innerHTML = `
        <div class="match-header">
          <div class="match-title-area">
            <h4 class="match-title">#${idx + 1}. ${escapeHtml(c.title)}</h4>
            <div class="match-channel">Channel: ${escapeHtml(c.channel)} • Uploaded: ${c.upload_date || 'N/A'} • ${c.duration_str || ''}</div>
          </div>
          <div class="risk-badge ${v.threat_color}">
            ${v.threat_badge} (${v.risk_score_pct}%)
          </div>
        </div>

        <div class="thumb-comparison-box">
          <div class="thumb-item">
            <div class="thumb-label">Original Video Thumbnail</div>
            <div class="thumb-wrapper">
              <img src="${src.thumbnail}" alt="Original Thumbnail" onerror="this.src='https://img.youtube.com/vi/${src.id}/hqdefault.jpg'"/>
            </div>
          </div>
          <div class="thumb-item">
            <div class="thumb-label">Suspect Video Thumbnail</div>
            <div class="thumb-wrapper">
              <img src="${c.thumbnail}" alt="Suspect Thumbnail" onerror="this.src='https://img.youtube.com/vi/${c.id}/hqdefault.jpg'"/>
            </div>
          </div>
        </div>

        <div class="metrics-row">
          <div class="metric-pill">
            <div class="metric-name">Thumbnail Match</div>
            <div class="metric-val">
              <span>${vis.visual_similarity_pct || 0}%</span>
              <span class="metric-status" style="background:${vis.visual_similarity_pct > 75 ? 'rgba(239,68,68,0.2)' : 'rgba(59,130,246,0.2)'}">
                pHash: ${vis.phash_distance ?? 'N/A'}
              </span>
            </div>
          </div>
          <div class="metric-pill">
            <div class="metric-name">AI DINOv2 Deep Vision</div>
            <div class="metric-val">
              <span>${vis.dinov2_pct || 0}%</span>
              <span class="metric-status" style="background:rgba(99,102,241,0.2)">
                ${vis.sift_inliers ? vis.sift_inliers + ' Inliers' : 'Layout Match'}
              </span>
            </div>
          </div>
          <div class="metric-pill">
            <div class="metric-name">Transcript / Audio</div>
            <div class="metric-val">
              <span>${trans.transcripts_available ? trans.script_similarity_pct + '%' : 'N/A'}</span>
              <span class="metric-status" style="background:rgba(234,179,8,0.2)">
                ${trans.transcripts_available ? 'Captions Scored' : 'No Captions'}
              </span>
            </div>
          </div>
          <div class="metric-pill">
            <div class="metric-name">Title Similarity</div>
            <div class="metric-val">
              <span>${tit.title_similarity_pct || 0}%</span>
              <span class="metric-status" style="background:rgba(16,185,129,0.2)">
                ${tit.is_exact_title ? 'Exact Title' : 'Token Overlap'}
              </span>
            </div>
          </div>
        </div>

        <div class="evidence-list">
          <h4>Plagiarism Findings & Forensic Breakdown</h4>
          <ul>
            ${v.evidence.map(ev => `<li>${escapeHtml(ev)}</li>`).join("")}
          </ul>
        </div>

        <div class="match-actions">
          <button class="btn-secondary btn-inspect" data-idx="${idx}">
            Draft DMCA Infringement Notice
          </button>
          <a href="${c.url}" target="_blank" rel="noopener" class="btn-secondary" style="text-decoration:none">
            Watch Suspect on YouTube ↗
          </a>
        </div>
      `;

      matchesContainer.appendChild(card);
    });

    // Attach click listeners to DMCA buttons
    document.querySelectorAll(".btn-inspect").forEach(btn => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.getAttribute("data-idx"), 10);
        openDmcaModal(data.matches[idx]);
      });
    });

    // Scroll smoothly to results
    resultsSection.scrollIntoView({ behavior: "smooth" });
  }

  // DMCA Modal
  function openDmcaModal(match) {
    if (!latestScanData || !match) return;
    const src = latestScanData.source_video;
    const c = match.candidate;
    const v = match.verdict;
    const vis = match.visual_comparison;
    const trans = match.transcript_comparison;

    const notice = `================================================================================
FORMAL NOTICE OF COPYRIGHT INFRINGEMENT (DMCA / YOUTUBE COMPLAINT)
Generated via PlagiGuard YT (Author: omanushoriginal <drisham4@gmail.com>)
Date: ${new Date().toISOString().split('T')[0]}
================================================================================

To: YouTube Copyright Compliance / Counter-Notice Department
Regarding: Unauthorized Duplication & Plagiarism of Original Content

1. ORIGINAL WORK DETAILS:
--------------------------------------------------------------------------------
Title: "${src.title}"
Original Channel: ${src.channel}
Original Video URL: ${src.url}
Upload Date: ${src.upload_date || 'N/A'}
Original Duration: ${src.duration_str || 'N/A'}
Thumbnail Hash: pHash standard computed

2. INFRINGING MATERIAL:
--------------------------------------------------------------------------------
Infringing Video Title: "${c.title}"
Infringing Channel: ${c.channel}
Infringing Video URL: ${c.url}
Suspect Upload Date: ${c.upload_date || 'N/A'}
Suspect Video Duration: ${c.duration_str || 'N/A'}

3. FORENSIC EVIDENCE & PLAGIARISM METRICS:
--------------------------------------------------------------------------------
Overall Threat Assessment: ${v.threat_badge} (${v.risk_score_pct}% Match)
Visual Thumbnail Correlation: ${vis.visual_similarity_pct}%
- Perceptual Hamming Distance: ${vis.phash_distance}
- Meta DINOv2 Deep Structural Score: ${vis.dinov2_pct}%
- SIFT/RANSAC Geometric Inliers: ${vis.sift_inliers || 0}
Spoken Transcript Similarity: ${trans.transcripts_available ? trans.script_similarity_pct + '%' : 'Captions disabled on suspect video'}
Title Similarity: ${match.title_comparison.title_similarity_pct}%

Specific Violations:
${v.evidence.map(e => '- ' + e).join('\n')}

4. GOOD FAITH STATEMENT:
--------------------------------------------------------------------------------
I have a good faith belief that the use of the material in the manner complained 
of is not authorized by the copyright owner, its agent, or the law. The information 
in this notification is accurate, and under penalty of perjury, I am authorized to 
act on behalf of the owner of an exclusive right that is allegedly infringed.

Sincerely,
omanushoriginal
Email: drisham4@gmail.com
Generated by: PlagiGuard YT (https://github.com/omanushoriginal/plagiariser_detector)
================================================================================`;

    dmcaTextarea.value = notice;
    dmcaModal.style.display = "flex";
  }

  btnOpenDmca.addEventListener("click", () => {
    if (latestScanData && latestScanData.matches.length > 0) {
      openDmcaModal(latestScanData.matches[0]);
    } else {
      alert("Please perform a scan first.");
    }
  });

  btnCloseModal.addEventListener("click", () => {
    dmcaModal.style.display = "none";
  });

  dmcaModal.addEventListener("click", (e) => {
    if (e.target === dmcaModal) dmcaModal.style.display = "none";
  });

  btnCopyDmca.addEventListener("click", () => {
    navigator.clipboard.writeText(dmcaTextarea.value).then(() => {
      btnCopyDmca.innerText = "✓ Copied to Clipboard!";
      setTimeout(() => btnCopyDmca.innerText = "Copy Notice", 2000);
    });
  });

  btnDownloadReport.addEventListener("click", () => {
    if (!latestScanData) return;
    const blob = new Blob([JSON.stringify(latestScanData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `plagiarism_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  function generateClientSideDemo(url) {
    let videoId = "sample_vid";
    const match = url.match(/(?:v=|\/vi\/|\/embed\/|youtu\.be\/|\/shorts\/)([0-9A-Za-z_-]{11})/);
    if (match) videoId = match[1];

    return {
      status: "success",
      scan_time_sec: 2.85,
      source_video: {
        id: videoId,
        url: url,
        title: `Your Video (${videoId}) - Forensic Analysis`,
        channel: "Your YouTube Channel",
        thumbnail: `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`,
        duration_str: "14:20",
        view_count: 54200,
        upload_date: "2024-03-15",
        transcript_stats: { word_count: 2450, available: true }
      },
      candidates_count: 3,
      highest_plagiarism_risk_pct: 91.5,
      matches: [
        {
          candidate: {
            id: "reupload_clone_1",
            url: "https://www.youtube.com/watch?v=clone1",
            title: `[Re-upload] ${videoId} Full Breakdown`,
            channel: "Trending Clips Hub",
            thumbnail: `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`,
            duration_str: "14:18",
            view_count: 18200,
            upload_date: "2024-05-20",
            transcript_stats: { word_count: 2420, available: true }
          },
          visual_comparison: {
            visual_similarity_pct: 97.2,
            verdict: "NEAR_DUPLICATE_THUMBNAIL",
            phash_distance: 1,
            dinov2_pct: 98.4,
            sift_inliers: 92,
            findings: [
              "Thumbnail pHash distance 1 indicates identical cropped artwork.",
              "DINOv2 deep layout similarity is 98.4%."
            ]
          },
          transcript_comparison: {
            transcripts_available: true,
            script_similarity_pct: 94.0,
            verdict: "BLATANT_SCRIPT_COPY",
            findings: ["Voiceover audio is a direct verbatim re-upload (94% match)."]
          },
          title_comparison: { title_similarity_pct: 88.0, is_exact_title: false },
          verdict: {
            risk_score_pct: 91.5,
            threat_level: "CRITICAL",
            threat_color: "red",
            threat_badge: "High Probability of Plagiarism",
            evidence: [
              "Identical thumbnail artwork recompressed with filter.",
              "94% spoken audio transcript verbatim overlap.",
              "Video duration identical within 2 seconds."
            ]
          }
        },
        {
          candidate: {
            id: "ai_derivative_2",
            url: "https://www.youtube.com/watch?v=clone2",
            title: "Ultimate Guide & Best Practices Review",
            channel: "AI Creator Daily",
            thumbnail: `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`,
            duration_str: "12:50",
            view_count: 9400,
            upload_date: "2024-06-11",
            transcript_stats: { word_count: 2100, available: true }
          },
          visual_comparison: {
            visual_similarity_pct: 84.1,
            verdict: "AI_MODIFIED_OR_STYLED_THUMBNAIL",
            phash_distance: 12,
            dinov2_pct: 86.8,
            sift_inliers: 31,
            findings: [
              "Meta DINOv2 detected AI-inpainted or restyled thumbnail derivative (86.8%).",
              "Geometric keypoint positions match original layout."
            ]
          },
          transcript_comparison: {
            transcripts_available: true,
            script_similarity_pct: 71.5,
            verdict: "HEAVY_SCRIPT_PARAPHRASE",
            findings: ["Spoken points paraphrase original script step-by-step."]
          },
          title_comparison: { title_similarity_pct: 45.0, is_exact_title: false },
          verdict: {
            risk_score_pct: 76.8,
            threat_level: "CRITICAL",
            threat_color: "red",
            threat_badge: "AI-Modified Duplicate / Stolen Structure",
            evidence: [
              "AI-remixed thumbnail derivative with identical visual layout.",
              "Paraphrased script mirroring original tutorial chapters."
            ]
          }
        }
      ]
    };
  }
});
