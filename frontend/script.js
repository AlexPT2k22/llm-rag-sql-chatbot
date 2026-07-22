document.addEventListener("DOMContentLoaded", () => {
  const chatContainer = document.getElementById("chat-container");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const welcomeMessage = document.getElementById("welcome-message");
  const sidebarCollapse = document.querySelector(".sidebar-collapse");
  const sidebar = document.querySelector(".sidebar");
  const API_BASE = "http://127.0.0.1:8000";

  marked.setOptions({ breaks: false, gfm: true });

  function fixMarkdownLists(text) {
    text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

    const lines = text.split('\n');

    for (let i = 1; i < lines.length; i++) {
      const prev = lines[i - 1].trim();
      const curr = lines[i].trim();
      const currIsTable = curr.startsWith('|');
      const prevIsTable = prev.startsWith('|');
      const currIsList = /^([-*+]|\d+\.)\s/.test(curr);
      const prevIsList = /^([-*+]|\d+\.)\s/.test(prev);

      if (currIsTable && prev !== '' && !prevIsTable) {
        lines[i] = '\n' + lines[i];
      }

      if (currIsList && prev !== '' && !prevIsList) {

        if (lines[i - 1].endsWith('  ')) {
          lines[i] = '\n' + lines[i];
        }
      }
    }

    return lines.join('\n');
  }

  function renderMarkdown(text) {
    const fixed = fixMarkdownLists(text);
    return marked.parse(fixed);
  }

  function addMessage(content, isUser, category) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${isUser ? 'message--user' : 'message--bot'}`;
    wrapper.innerHTML = `
      <div class="message__avatar">
        <i class="ph ph-${isUser ? 'user' : 'robot'}"></i>
      </div>
      <div class="message__content">
        ${category && !isUser ? `<span class="message__category">${category}</span>` : ''}
        <div class="message__text">${isUser ? escapeHtml(content) : renderMarkdown(content)}</div>
      </div>
    `;
    return wrapper;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function startBotMessage(category) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message message--bot';
    wrapper.innerHTML = `
      <div class="message__avatar">
        <i class="ph ph-robot"></i>
      </div>
      <div class="message__content">
        <span class="message__category">${category}</span>
        <div class="message__text" id="streaming-text"></div>
      </div>
    `;
    return wrapper;
  }

  let eventSource = null;

  function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    userInput.value = '';
    sendBtn.disabled = true;
    userInput.style.height = 'auto';

    if (welcomeMessage) welcomeMessage.remove();

    const userMsg = addMessage(message, true);
    chatContainer.appendChild(userMsg);

    const botMsg = startBotMessage('...');
    chatContainer.appendChild(botMsg);

    const textEl = botMsg.querySelector('.message__text');
    const categoryEl = botMsg.querySelector('.message__category');

    if (eventSource) eventSource.close();

    const entidade_id = sessionStorage.getItem('entidade_id');
    const ano_agricola_id = sessionStorage.getItem('ano_agricola_id');

    let url = `${API_BASE}/chat/stream?message=${encodeURIComponent(message)}&session_id=${sessionId}`;
    if (entidade_id) url += `&entidade_id=${entidade_id}`;
    if (ano_agricola_id) url += `&ano_agricola_id=${ano_agricola_id}`;

    eventSource = new EventSource(url);
    let accumulated = '';

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'category' && data.category) {
        categoryEl.textContent = data.category;
      }

      if (data.type === 'token' && data.token) {
        accumulated += data.token;
        textEl.innerHTML = renderMarkdown(accumulated);
        scrollToBottom();
      }

      if (data.type === 'done') {
        eventSource.close();
        eventSource = null;
        if (data.sql_query) {
          console.log('SQL:', data.sql_query);
        }
      }
    };

    eventSource.onerror = () => {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      if (textEl.textContent.trim() === '') {
        textEl.textContent = 'Erro de ligacao. Tente novamente.';
      }
    };

    scrollToBottom();
  }

  function scrollToBottom() {
    const container = document.getElementById('chat-container');
    if (container) container.scrollTop = container.scrollHeight;
  }

  sendBtn.addEventListener('click', sendMessage);

  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
    sendBtn.disabled = !userInput.value.trim();
  });

  sidebarCollapse.addEventListener('click', () => {
    sidebar.classList.toggle('sidebar--collapsed');
    document.body.classList.toggle('sidebar-collapsed');
  });

  document.getElementById('new-chat-btn').addEventListener('click', () => {
    sessionId = crypto.randomUUID();
    chatContainer.innerHTML = `
      <div class="welcome-container" id="welcome-message">
        <h1>Como posso ajudar?</h1>
        <p>Faca perguntas sobre os dados dos modulos ou sobre os procedimentos e documentacao funcional.</p>
      </div>
    `;
  });

  document.getElementById('logout-btn').addEventListener('click', () => {
    sessionStorage.clear();
    window.location.href = 'login.html';
  });

  const entidadeLabel = sessionStorage.getItem('ano_agricola_label') || '';
  const entidadeId = sessionStorage.getItem('entidade_id') || '';
  if (entidadeId) {
    document.getElementById('entidade-info').textContent = `Entidade ${entidadeId} ${entidadeLabel}`;
  }

  let sessionId = crypto.randomUUID();
});
