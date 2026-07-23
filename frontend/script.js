(function () {
  if (!sessionStorage.getItem("entidade_id")) {
    window.location.href = "login.html";
    return;
  }

  const chatContainer = document.getElementById("chat-container");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const sidebarCollapse = document.querySelector(".sidebar-collapse");
  const sidebar = document.querySelector(".sidebar");
  const chatHistoryList = document.getElementById("chat-history-list");
  const overlay = document.getElementById("sidebar-overlay");
  const menuBtn = document.getElementById("menu-btn");

  if (!chatContainer || !userInput || !sendBtn) return;

  marked.setOptions({ breaks: false, gfm: true });

  /* ── Session State ── */

  let sessionId = crypto.randomUUID();
  let eventSource = null;
  let currentTraceId = null;
  let currentBotWrapper = null;
  let generating = false;

  const STORAGE_KEY = "agrisystem_sessions";
  const STORAGE_ORDER = "agrisystem_session_order";
  const STORAGE_CURRENT = "agrisystem_current_session";
  const STORAGE_FEEDBACK = "agrisystem_feedback";
  const STORAGE_SIDEBAR = "agrisystem_sidebar_collapsed";
  const MAX_SESSIONS = 20;

  function getSessions() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch { return {}; }
  }
  function getSessionOrder() {
    try { return JSON.parse(localStorage.getItem(STORAGE_ORDER)) || []; } catch { return []; }
  }
  function saveSessions(s) { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); } catch {} }
  function saveSessionOrder(o) { try { localStorage.setItem(STORAGE_ORDER, JSON.stringify(o)); } catch {} }
  function getFeedbackState() {
    try { return JSON.parse(localStorage.getItem(STORAGE_FEEDBACK)) || {}; } catch { return {}; }
  }
  function setFeedbackState(tid, score, comment) {
    const s = getFeedbackState();
    s[tid] = { score, comment, ts: Date.now() };
    try { localStorage.setItem(STORAGE_FEEDBACK, JSON.stringify(s)); } catch {}
  }

  /* ── Persistence ── */

  function persistCurrentSession() {
    const messages = [];
    chatContainer.querySelectorAll(".message").forEach((w) => {
      const isUser = w.classList.contains("message--user");
      const textEl = w.querySelector(".message__text");
      const catEl = w.querySelector(".message__category");
      if (textEl) {
        messages.push({
          role: isUser ? "user" : "bot",
          content: isUser ? textEl.textContent : textEl.innerHTML,
          category: catEl ? catEl.textContent.replace("Category: ", "") : null,
          traceId: w.dataset.traceId || null,
        });
      }
    });
    if (messages.length === 0) return;
    const sessions = getSessions();
    const order = getSessionOrder();
    const title = (messages.find((m) => m.role === "user") || {}).content || "Nova Conversa";
    sessions[sessionId] = { id: sessionId, title: title.length > 60 ? title.slice(0, 60) + "\u2026" : title, messages, updated: Date.now() };
    if (!order.includes(sessionId)) order.unshift(sessionId);
    else { const i = order.indexOf(sessionId); order.splice(i, 1); order.unshift(sessionId); }
    while (order.length > MAX_SESSIONS) { const old = order.pop(); delete sessions[old]; }
    saveSessions(sessions);
    saveSessionOrder(order);
    localStorage.setItem(STORAGE_CURRENT, sessionId);
    renderSessionList();
  }

  function loadSession(id) {
    const sessions = getSessions();
    const session = sessions[id];
    if (!session || !session.messages) return;
    sessionId = id;
    localStorage.setItem(STORAGE_CURRENT, sessionId);
    chatContainer.innerHTML = "";
    session.messages.forEach((msg) => {
      if (msg.role === "user") {
        chatContainer.appendChild(addMessageElement(msg.content, true, null));
      } else {
        const el = addMessageElement(msg.content, false, msg.category);
        if (msg.traceId) el.dataset.traceId = msg.traceId;
        addCopyButtons(el);
        addFeedbackRow(el, msg.traceId);
        chatContainer.appendChild(el);
      }
    });
    scrollToBottom();
    renderSessionList();
  }

  function newSession() {
    persistCurrentSession();
    sessionId = crypto.randomUUID();
    localStorage.setItem(STORAGE_CURRENT, sessionId);
    chatContainer.innerHTML = renderWelcomeHTML();
    renderSessionList();
  }

  function deleteSession(id, e) {
    e.stopPropagation();
    if (!confirm("Apagar esta conversa?")) return;
    const sessions = getSessions();
    const order = getSessionOrder().filter((oid) => oid !== id);
    delete sessions[id];
    saveSessions(sessions);
    saveSessionOrder(order);
    if (sessionId === id) newSession();
    else renderSessionList();
  }

  function renderSessionList() {
    if (!chatHistoryList) return;
    const order = getSessionOrder();
    const sessions = getSessions();
    if (order.length === 0 || Object.keys(sessions).length === 0) {
      chatHistoryList.innerHTML = '<span class="nav-empty">Sem conversas anteriores</span>';
      return;
    }
    chatHistoryList.innerHTML = "";
    order.forEach((id) => {
      const s = sessions[id];
      if (!s) return;
      const wrapper = document.createElement("a");
      wrapper.className = "nav-item" + (id === sessionId ? " active" : "");
      wrapper.href = "#";
      const titleSpan = document.createElement("span");
      titleSpan.className = "nav-item-title";
      titleSpan.textContent = s.title || "Nova Conversa";
      const delBtn = document.createElement("button");
      delBtn.className = "nav-item-delete";
      delBtn.innerHTML = '<i class="ph ph-x"></i>';
      delBtn.title = "Apagar conversa";
      delBtn.addEventListener("click", (e) => deleteSession(id, e));
      delBtn.addEventListener("mousedown", (e) => e.stopPropagation());
      wrapper.appendChild(titleSpan);
      wrapper.appendChild(delBtn);
      wrapper.addEventListener("click", (e) => {
        e.preventDefault();
        if (id !== sessionId) { persistCurrentSession(); loadSession(id); }
      });
      chatHistoryList.appendChild(wrapper);
    });
  }

  /* ── Markdown ── */

  function fixMarkdownLists(text) {
    text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const lines = text.split("\n");
    for (let i = 1; i < lines.length; i++) {
      const prev = lines[i - 1].trim();
      const curr = lines[i].trim();
      const currIsTable = curr.startsWith("|");
      const prevIsTable = prev.startsWith("|");
      const currIsList = /^([-*+]|\d+\.)\s/.test(curr);
      const prevIsList = /^([-*+]|\d+\.)\s/.test(prev);
      if (currIsTable && prev !== "" && !prevIsTable) lines[i] = "\n" + lines[i];
      if (currIsList && prev !== "" && !prevIsList && lines[i - 1].endsWith("  ")) lines[i] = "\n" + lines[i];
    }
    return lines.join("\n");
  }

  function renderMarkdown(text) { return marked.parse(fixMarkdownLists(text)); }
  function escapeHtml(text) { const d = document.createElement("div"); d.textContent = text; return d.innerHTML; }

  /* ── Welcome ── */

  function renderWelcomeHTML() {
    return '<div class="welcome-container" id="welcome-message">'
      + '<h1>How can I help?</h1>'
      + '<p>Ask questions about module data or about procedures and functional documentation.</p>'
      + '<div class="suggestion-chips">'
      + '<button class="suggestion-chip" data-question="How many active plots are there?"><i class="ph ph-tree"></i> Active plots</button>'
      + '<button class="suggestion-chip" data-question="What is the procedure for registering a harvest?"><i class="ph ph-wine"></i> Harvest procedure</button>'
      + '<button class="suggestion-chip" data-question="List the operations from last week"><i class="ph ph-calendar-check"></i> Recent operations</button>'
      + '<button class="suggestion-chip" data-question="What are the most used plant protection products?"><i class="ph ph-flask"></i> Plant protection products</button>'
      + '</div></div>';
  }

  /* ── Category Color ── */

  function getCategoryClass(cat) {
    if (!cat) return "";
    const c = cat.toLowerCase();
    if (c === "sql") return "message__category--sql";
    if (c === "rag") return "message__category--rag";
    if (c === "both") return "message__category--both";
    if (c === "meta") return "message__category--meta";
    if (c === "greeting" || c === "chitchat") return "message__category--" + c;
    return "";
  }

  /* ── Message Elements ── */

  function addMessageElement(content, isUser, category) {
    const wrapper = document.createElement("div");
    wrapper.className = "message " + (isUser ? "message--user" : "message--bot");
    const catClass = !isUser && category ? getCategoryClass(category) : "";
    const catLabel = !isUser && category ? category : "";
    wrapper.innerHTML = '<div class="message__avatar"><i class="ph ph-' + (isUser ? "user" : "robot") + '"></i></div>'
      + '<div class="message__content">'
      + (catLabel ? '<span class="message__category ' + catClass + '">' + 'Category: ' + catLabel + '</span>' : '')
      + '<div class="message__text">' + (isUser ? escapeHtml(content) : renderMarkdown(content)) + '</div>'
      + '</div>';
    return wrapper;
  }

  function addCopyButtons(wrapper) {
    wrapper.querySelectorAll(".message__text pre").forEach((pre) => {
      if (pre.querySelector(".copy-btn")) return;
      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.innerHTML = '<i class="ph ph-copy"></i> <span>Copiar</span>';
      btn.addEventListener("click", () => {
        navigator.clipboard.writeText(pre.textContent || "").then(() => {
          btn.classList.add("copied");
          btn.querySelector("span").textContent = "Copiado!";
          setTimeout(() => { btn.classList.remove("copied"); btn.querySelector("span").textContent = "Copiar"; }, 2000);
        }).catch(() => {});
      });
      pre.style.position = "relative";
      pre.appendChild(btn);
    });
  }

  function addFeedbackRow(wrapper, traceId) {
    if (!traceId || wrapper.querySelector(".feedback-row")) return;
    const msgContent = wrapper.querySelector(".message__content");
    if (!msgContent) return;
    const row = document.createElement("div");
    row.className = "feedback-row";
    const likeBtn = document.createElement("button");
    likeBtn.className = "feedback-btn";
    likeBtn.innerHTML = '<i class="ph ph-thumbs-up"></i>';
    likeBtn.title = "Resposta \u00fatil";
    const dislikeBtn = document.createElement("button");
    dislikeBtn.className = "feedback-btn";
    dislikeBtn.innerHTML = '<i class="ph ph-thumbs-down"></i>';
    dislikeBtn.title = "Resposta incorreta";
    const existing = getFeedbackState()[traceId];
    if (existing) {
      if (existing.score === 1) { likeBtn.classList.add("selected-like"); likeBtn.disabled = true; dislikeBtn.disabled = true; }
      else if (existing.score === 0) { dislikeBtn.classList.add("selected-dislike"); likeBtn.disabled = true; dislikeBtn.disabled = true; }
    }
    likeBtn.addEventListener("click", () => { sendFeedback(traceId, 1, ""); likeBtn.classList.add("selected-like"); likeBtn.disabled = true; dislikeBtn.disabled = true; });
    dislikeBtn.addEventListener("click", () => { showFeedbackModal(traceId, () => { dislikeBtn.classList.add("selected-dislike"); likeBtn.disabled = true; dislikeBtn.disabled = true; }); });
    row.appendChild(likeBtn);
    row.appendChild(dislikeBtn);
    msgContent.appendChild(row);
  }

  function sendFeedback(traceId, score, comment) {
    if (!traceId) return;
    const fd = new FormData();
    fd.append("trace_id", traceId);
    fd.append("score", score);
    fd.append("comment", comment);
    fetch(API_BASE + "/feedback", { method: "POST", body: fd })
      .then(() => setFeedbackState(traceId, score, comment))
      .catch(() => toast("Erro ao enviar feedback", "error"));
  }

  function showFeedbackModal(traceId, onSent) {
    const existing = document.querySelector(".feedback-modal-overlay");
    if (existing) existing.remove();
    const overlay = document.createElement("div");
    overlay.className = "feedback-modal-overlay";
    overlay.innerHTML = '<div class="feedback-modal"><h3>O que estava errado?</h3>'
      + '<textarea id="feedback-comment" placeholder="Descreva o problema..."></textarea>'
      + '<div class="feedback-modal-actions">'
      + '<button id="feedback-cancel">Cancelar</button>'
      + '<button id="feedback-submit" class="primary">Enviar</button>'
      + '</div></div>';
    document.body.appendChild(overlay);
    const textarea = overlay.querySelector("#feedback-comment");
    const cancelBtn = overlay.querySelector("#feedback-cancel");
    const submitBtn = overlay.querySelector("#feedback-submit");
    function close() { overlay.remove(); }
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    cancelBtn.addEventListener("click", close);
    submitBtn.addEventListener("click", () => { sendFeedback(traceId, 0, textarea.value.trim() || "dislike"); close(); if (onSent) onSent(); });
    textarea.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    setTimeout(() => textarea.focus(), 100);
  }

  /* ── Toast ── */

  function toast(message, type) {
    type = type || "info";
    let container = document.querySelector(".toast-container");
    if (!container) { container = document.createElement("div"); container.className = "toast-container"; document.body.appendChild(container); }
    const iconMap = { success: "check-circle", error: "warning-circle", info: "info" };
    const el = document.createElement("div");
    el.className = "toast toast--" + type;
    el.innerHTML = '<i class="ph ph-' + (iconMap[type] || "info") + '"></i> ' + escapeHtml(message);
    container.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; el.style.transform = "translateX(24px)"; el.style.transition = "all 0.25s ease"; setTimeout(() => el.remove(), 250); }, 3500);
  }

  /* ── Streaming ── */

  function startBotMessage(category) {
    const wrapper = document.createElement("div");
    wrapper.className = "message message--bot";
    const catClass = getCategoryClass(category);
    wrapper.innerHTML = '<div class="message__avatar"><i class="ph ph-robot"></i></div>'
      + '<div class="message__content">'
      + '<span class="message__category ' + catClass + '">' + category + '</span>'
      + '<div class="message__text"><div class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div></div>'
      + '</div>';
    return wrapper;
  }

  function setGenerating(active) {
    generating = active;
    if (active) {
      sendBtn.classList.add("stop");
      sendBtn.disabled = false;
      sendBtn.title = "Parar gera\u00e7\u00e3o";
      sendBtn.querySelector("i").className = "ph ph-stop";
    } else {
      sendBtn.classList.remove("stop");
      sendBtn.title = "Enviar mensagem";
      sendBtn.querySelector("i").className = "ph ph-arrow-up";
      sendBtn.disabled = !userInput.value.trim();
    }
  }

  function stopGeneration() {
    if (eventSource) { eventSource.close(); eventSource = null; }
    setGenerating(false);
    if (currentBotWrapper) {
      const textEl = currentBotWrapper.querySelector(".message__text");
      if (textEl) {
        const dots = textEl.querySelector(".typing-indicator");
        if (dots) dots.remove();
        if (!textEl.textContent.trim()) textEl.innerHTML = "[gera\u00e7\u00e3o interrompida]";
        else textEl.innerHTML += " <em>[gera\u00e7\u00e3o interrompida]</em>";
      }
      if (currentTraceId) { currentBotWrapper.dataset.traceId = currentTraceId; addCopyButtons(currentBotWrapper); addFeedbackRow(currentBotWrapper, currentTraceId); }
    }
    persistCurrentSession();
    currentBotWrapper = null;
    currentTraceId = null;
    userInput.focus();
  }

  function sendMessage() {
    if (generating) { stopGeneration(); return; }
    const message = userInput.value.trim();
    if (!message) return;
    userInput.value = "";
    userInput.style.height = "auto";
    sendBtn.disabled = true;
    const welcomeMsg = document.getElementById("welcome-message");
    if (welcomeMsg) welcomeMsg.remove();
    const userMsg = addMessageElement(message, true, null);
    chatContainer.appendChild(userMsg);
    const botMsg = startBotMessage("...");
    chatContainer.appendChild(botMsg);
    currentBotWrapper = botMsg;
    currentTraceId = null;
    setGenerating(true);
    const textEl = botMsg.querySelector(".message__text");
    const categoryEl = botMsg.querySelector(".message__category");
    if (eventSource) { eventSource.close(); eventSource = null; }
    const entidadeId = sessionStorage.getItem("entidade_id");
    const anoAgricolaId = sessionStorage.getItem("ano_agricola_id");
    let url = API_BASE + "/chat/stream?message=" + encodeURIComponent(message) + "&session_id=" + sessionId;
    if (entidadeId) url += "&entidade_id=" + entidadeId;
    if (anoAgricolaId) url += "&ano_agricola_id=" + anoAgricolaId;
    eventSource = new EventSource(url);
    let accumulated = "";
    let firstTokenReceived = false;
    let capturedSql = null;

    eventSource.onmessage = (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.type === "category" && data.category) {
        if (categoryEl) {
          categoryEl.textContent = data.category;
          categoryEl.className = "message__category " + getCategoryClass(data.category);
        }
      }
      if (data.type === "sql" && data.query) { capturedSql = data.query; }
      if (data.type === "token" && data.token) {
        if (!firstTokenReceived) { firstTokenReceived = true; const d = textEl.querySelector(".typing-indicator"); if (d) d.remove(); }
        accumulated += data.token;
        textEl.innerHTML = renderMarkdown(accumulated);
        scrollToBottom();
      }
      if (data.type === "done") {
        eventSource.close();
        eventSource = null;
        setGenerating(false);
        currentTraceId = data.trace_id || null;
        if (currentTraceId) botMsg.dataset.traceId = currentTraceId;
        addCopyButtons(botMsg);
        addFeedbackRow(botMsg, currentTraceId);
        if (capturedSql) {
          const sqlDiv = document.createElement("div");
          sqlDiv.className = "message__text";
          sqlDiv.style.cssText = "margin-top:12px;font-size:0.82rem;";
          sqlDiv.innerHTML = "<pre><code>" + escapeHtml(capturedSql) + "</code></pre>";
          addCopyButtons(sqlDiv);
          botMsg.querySelector(".message__content").appendChild(sqlDiv);
        }
        persistCurrentSession();
        currentBotWrapper = null;
      }
    };

    eventSource.onerror = () => {
      if (eventSource) { eventSource.close(); eventSource = null; }
      setGenerating(false);
      const dots = textEl.querySelector(".typing-indicator");
      if (dots) dots.remove();
      if (!textEl.textContent.trim()) textEl.innerHTML = "Erro de liga\u00e7\u00e3o. Tente novamente.";
      persistCurrentSession();
      currentBotWrapper = null;
    };
    scrollToBottom();
  }

  function scrollToBottom() { if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight; }

  /* ── Event Listeners ── */

  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
  userInput.addEventListener("input", () => { userInput.style.height = "auto"; userInput.style.height = userInput.scrollHeight + "px"; if (!generating) sendBtn.disabled = !userInput.value.trim(); });

  /* ── Sidebar Collapse ── */

  function applySidebarState() {
    const collapsed = localStorage.getItem(STORAGE_SIDEBAR) === "1";
    sidebar.classList.toggle("sidebar--collapsed", collapsed);
    document.body.classList.toggle("sidebar-collapsed", collapsed);
  }

  function isMobile() { return window.innerWidth <= 768; }

  if (sidebarCollapse) {
    sidebarCollapse.addEventListener("click", () => {
      if (isMobile()) {
        sidebar.classList.remove("open");
        overlay.classList.remove("sidebar-overlay--visible");
        document.body.style.overflow = "";
      } else {
        const isNow = sidebar.classList.toggle("sidebar--collapsed");
        document.body.classList.toggle("sidebar-collapsed", isNow);
        localStorage.setItem(STORAGE_SIDEBAR, isNow ? "1" : "0");
      }
    });
  }

  /* ── Mobile Hamburger ── */

  function openMobileSidebar() {
    sidebar.classList.add("open");
    overlay.classList.add("sidebar-overlay--visible");
    document.body.style.overflow = "hidden";
  }

  function closeMobileSidebar() {
    sidebar.classList.remove("open");
    overlay.classList.remove("sidebar-overlay--visible");
    document.body.style.overflow = "";
  }

  if (menuBtn) {
    menuBtn.addEventListener("click", () => {
      if (sidebar.classList.contains("open")) closeMobileSidebar();
      else openMobileSidebar();
    });
  }

  if (overlay) {
    overlay.addEventListener("click", closeMobileSidebar);
  }

  /* ── New Chat & Logout ── */

  const newChatBtn = document.getElementById("new-chat-btn");
  if (newChatBtn) {
    newChatBtn.addEventListener("click", () => {
      persistCurrentSession();
      newSession();
      closeMobileSidebar();
    });
  }

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      persistCurrentSession();
      sessionStorage.clear();
      window.location.href = "login.html";
    });
  }

  /* ── Entity Info ── */

  const entidadeId = sessionStorage.getItem("entidade_id") || "";
  const entidadeLabel = sessionStorage.getItem("ano_agricola_label") || "";
  const entidadeInfoEl = document.getElementById("entidade-info");
  if (entidadeInfoEl && entidadeId) {
    entidadeInfoEl.textContent = "Entidade " + entidadeId + " " + entidadeLabel;
  }

  /* ── Init ── */

  applySidebarState();
  const savedCurrent = localStorage.getItem(STORAGE_CURRENT);
  const sessions = getSessions();
  if (savedCurrent && sessions[savedCurrent] && sessions[savedCurrent].messages && sessions[savedCurrent].messages.length > 0) {
    sessionId = savedCurrent;
    loadSession(sessionId);
  } else {
    const welcomeEl = document.getElementById("welcome-message");
    if (welcomeEl) welcomeEl.outerHTML = renderWelcomeHTML();
    renderSessionList();
  }

  /* ── Suggestion Chips ── */

  chatContainer.addEventListener("click", (e) => {
    const chip = e.target.closest(".suggestion-chip");
    if (!chip) return;
    const question = chip.dataset.question;
    if (question) {
      userInput.value = question;
      userInput.style.height = "auto";
      userInput.style.height = userInput.scrollHeight + "px";
      sendBtn.disabled = false;
      sendMessage();
    }
  });

  /* ── Keyboard: Escape to stop ── */

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && generating) { e.preventDefault(); stopGeneration(); }
  });

})();
