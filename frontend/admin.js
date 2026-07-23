(function () {
  const uploadArea = document.getElementById("upload-area");
  const fileInput = document.getElementById("file-input");
  const uploadLabel = document.getElementById("upload-label");
  const uploadStatus = document.getElementById("upload-status");
  const uploadProgress = document.getElementById("upload-progress");
  const progressBar = document.getElementById("progress-bar");
  const refreshBtn = document.getElementById("refresh-btn");
  const docsTbody = document.getElementById("docs-tbody");
  const docsFooter = document.getElementById("docs-footer");

  function extInfo(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    if (ext === "pdf") return { icon: "ph-file-pdf", label: "PDF" };
    if (ext === "md") return { icon: "ph-file-md", label: "Markdown" };
    if (ext === "txt") return { icon: "ph-file-text", label: "Texto" };
    return { icon: "ph-file", label: ext.toUpperCase() };
  }

  function showStatus(msg, type) {
    uploadStatus.textContent = msg;
    uploadStatus.className = "upload-status upload-status--" + type;
    uploadStatus.hidden = false;
    if (type === "success") setTimeout(function () { uploadStatus.hidden = true; }, 5000);
  }

  async function loadDocuments() {
    docsTbody.innerHTML = '<tr><td colspan="4" class="docs-state">A carregar...</td></tr>';

    try {
      const resp = await fetch(API_BASE + "/admin/documents");
      if (!resp.ok) throw new Error("Falha ao carregar documentos");
      const data = await resp.json();
      if (data.documents.length === 0) {
        docsTbody.innerHTML = '<tr><td colspan="4" class="docs-state">Nenhum documento indexado</td></tr>';
        docsFooter.textContent = "0 documentos";
        return;
      }
      docsFooter.textContent = data.documents.length + " documentos";
      docsTbody.innerHTML = data.documents.map(function (d) {
        var info = extInfo(d.name);
        return '<tr>' +
          '<td><i class="ph ' + info.icon + ' docs-icon"></i> ' + d.name + '</td>' +
          '<td class="col-type">' + info.label + '</td>' +
          '<td class="col-size">' + d.size_kb + ' KB</td>' +
          '<td class="col-actions"><button class="delete-btn" data-file="' + d.name + '" title="Remover"><i class="ph ph-trash"></i></button></td>' +
          '</tr>';
      }).join("");

      docsTbody.querySelectorAll(".delete-btn").forEach(function (btn) {
        btn.addEventListener("click", async function () {
          var filename = btn.dataset.file;
          if (!confirm('Remover "' + filename + '"?')) return;
          try {
            var resp = await fetch(API_BASE + "/admin/documents/" + encodeURIComponent(filename), { method: "DELETE" });
            if (resp.ok) {
              showStatus('"' + filename + '" removido.', "success");
              loadDocuments();
            } else {
              showStatus('Erro ao remover "' + filename + '".', "error");
            }
          } catch (e) {
            showStatus('Erro de rede ao remover "' + filename + '".', "error");
          }
        });
      });
    } catch (err) {
      docsTbody.innerHTML = '<tr><td colspan="4" class="docs-state" style="color:#991b1b;">Erro: ' + err.message + '</td></tr>';
      docsFooter.textContent = "0 documentos";
    }
  }

  uploadArea.addEventListener("click", function () { fileInput.click(); });
  uploadLabel.addEventListener("click", function (e) { e.stopPropagation(); });

  uploadArea.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadArea.classList.add("upload-area--active");
  });

  uploadArea.addEventListener("dragleave", function () {
    uploadArea.classList.remove("upload-area--active");
  });

  uploadArea.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadArea.classList.remove("upload-area--active");
    handleFiles(e.dataTransfer.files);
  });

  fileInput.addEventListener("change", function () { handleFiles(fileInput.files); });

  async function handleFiles(files) {
    if (!files || files.length === 0) return;
    var file = files[0];
    var ext = file.name.split(".").pop().toLowerCase();
    if (["md", "txt", "pdf"].indexOf(ext) === -1) {
      showStatus("Formato nao suportado. Use .md, .txt ou .pdf.", "error");
      return;
    }
    var form = new FormData();
    form.append("file", file);

    uploadProgress.hidden = false;
    progressBar.style.width = "0%";

    try {
      var xhr = new XMLHttpRequest();
      xhr.open("POST", API_BASE + "/admin/upload");
      xhr.upload.onprogress = function (e) {
        if (e.lengthComputable) {
          progressBar.style.width = Math.round((e.loaded / e.total) * 100) + "%";
        }
      };
      xhr.onload = function () {
        uploadProgress.hidden = true;
        if (xhr.status === 200) {
          var data = JSON.parse(xhr.responseText);
          showStatus('"' + data.filename + '" indexado (' + data.chunks_indexed + ' chunks).', "success");
          loadDocuments();
          fileInput.value = "";
        } else {
          showStatus("Erro: " + xhr.responseText, "error");
        }
      };
      xhr.onerror = function () {
        uploadProgress.hidden = true;
        showStatus("Erro de rede.", "error");
      };
      xhr.send(form);
    } catch (err) {
      uploadProgress.hidden = true;
      showStatus("Erro: " + err.message, "error");
    }
  }

  refreshBtn.addEventListener("click", loadDocuments);

  loadDocuments();
})();
