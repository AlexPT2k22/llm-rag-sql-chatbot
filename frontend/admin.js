const API = "http://127.0.0.1:8000";

const uploadArea   = document.getElementById("upload-area");
const fileInput    = document.getElementById("file-input");
const uploadLabel  = document.getElementById("upload-label");
const uploadStatus = document.getElementById("upload-status");
const uploadProgress = document.getElementById("upload-progress");
const progressBar  = document.getElementById("progress-bar");
const refreshBtn   = document.getElementById("refresh-btn");
const docsTbody    = document.getElementById("docs-tbody");
const docsFooter   = document.getElementById("docs-footer");

function extIcon(name) {
  const ext = name.split(".").pop().toLowerCase();
  if (ext === "pdf")  return "ph-file-pdf";
  if (ext === "md")   return "ph-file-md";
  return "ph-file-text";
}

function showStatus(msg, type) {
  uploadStatus.textContent = msg;
  uploadStatus.className = `upload-status upload-status--${type}`;
  uploadStatus.hidden = false;
  if (type === "success") setTimeout(() => { uploadStatus.hidden = true; }, 5000);
}

async function loadDocuments() {
  docsTbody.innerHTML = `<tr><td colspan="3" class="docs-state">A carregar...</td></tr>`;

  try {
    const resp = await fetch(`${API}/admin/documents`);
    if (!resp.ok) throw new Error("Falha ao carregar documentos");
    const data = await resp.json();
    if (data.documents.length === 0) {
      docsTbody.innerHTML = `<tr><td colspan="3" class="docs-state docs-state--empty">Nenhum documento indexado</td></tr>`;
      docsFooter.textContent = "0 documentos";
      return;
    }
    docsFooter.textContent = `${data.documents.length} documentos`;
    docsTbody.innerHTML = data.documents.map(d =>
      `<tr>
        <td><i class="ph ${extIcon(d.name)} docs-icon"></i> ${d.name}</td>
        <td>${d.size_kb} KB</td>
        <td><button class="btn-ico btn-ico--danger" data-file="${d.name}" title="Remover"><i class="ph ph-trash"></i></button></td>
      </tr>`
    ).join('');

    docsTbody.querySelectorAll('.btn-ico--danger').forEach(btn => {
      btn.addEventListener('click', async () => {
        const filename = btn.dataset.file;
        if (!confirm(`Remover "${filename}"?`)) return;
        try {
          const resp = await fetch(`${API}/admin/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
          if (resp.ok) {
            showStatus(`"${filename}" removido.`, 'success');
            loadDocuments();
          } else {
            showStatus(`Erro ao remover "${filename}".`, 'error');
          }
        } catch {
          showStatus(`Erro de rede ao remover "${filename}".`, 'error');
        }
      });
    });
  } catch (err) {
    docsTbody.innerHTML = `<tr><td colspan="3" class="docs-state docs-state--error">Erro: ${err.message}</td></tr>`;
    docsFooter.textContent = "0 documentos";
  }
}

uploadArea.addEventListener("click", () => fileInput.click());
uploadLabel.addEventListener("click", (e) => e.stopPropagation());

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("upload-area--active");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("upload-area--active");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("upload-area--active");
  handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => handleFiles(fileInput.files));

async function handleFiles(files) {
  if (!files || files.length === 0) return;
  const file = files[0];
  const ext = file.name.split(".").pop().toLowerCase();
  if (!["md", "txt", "pdf"].includes(ext)) {
    showStatus("Formato nao suportado. Use .md, .txt ou .pdf.", "error");
    return;
  }
  const form = new FormData();
  form.append("file", file);

  uploadProgress.hidden = false;
  progressBar.style.width = "0%";

  try {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/admin/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        progressBar.style.width = `${pct}%`;
      }
    };
    xhr.onload = () => {
      uploadProgress.hidden = true;
      if (xhr.status === 200) {
        const data = JSON.parse(xhr.responseText);
        showStatus(`"${data.filename}" indexado (${data.chunks_indexed} chunks).`, 'success');
        loadDocuments();
        fileInput.value = "";
      } else {
        showStatus(`Erro: ${xhr.responseText}`, 'error');
      }
    };
    xhr.onerror = () => {
      uploadProgress.hidden = true;
      showStatus("Erro de rede.", "error");
    };
    xhr.send(form);
  } catch (err) {
    uploadProgress.hidden = true;
    showStatus(`Erro: ${err.message}`, "error");
  }
}

refreshBtn.addEventListener("click", loadDocuments);

loadDocuments();
