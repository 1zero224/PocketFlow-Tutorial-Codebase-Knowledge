const state = {
  selectedPath: "",
  pollHandle: null,
  defaults: null,
};

const elements = {
  queueMode: document.getElementById("queue-mode"),
  activeTask: document.getElementById("active-task"),
  startQueueButton: document.getElementById("start-queue-button"),
  pickFolderButton: document.getElementById("pick-folder-button"),
  currentPathLabel: document.getElementById("current-path-label"),
  taskForm: document.getElementById("task-form"),
  sourcePath: document.getElementById("source-path"),
  projectName: document.getElementById("project-name"),
  outputDir: document.getElementById("output-dir"),
  language: document.getElementById("language"),
  useCache: document.getElementById("use-cache"),
  maxFileSize: document.getElementById("max-file-size"),
  maxAbstractions: document.getElementById("max-abstractions"),
  maxBatches: document.getElementById("max-batches"),
  llmConcurrency: document.getElementById("llm-concurrency"),
  includePatterns: document.getElementById("include-patterns"),
  excludePatterns: document.getElementById("exclude-patterns"),
  formMessage: document.getElementById("form-message"),
  pendingCount: document.getElementById("pending-count"),
  completedCount: document.getElementById("completed-count"),
  failedCount: document.getElementById("failed-count"),
  taskList: document.getElementById("task-list"),
};

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function setMessage(text, isError = false) {
  elements.formMessage.textContent = text;
  elements.formMessage.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function setSelectedPath(pathText) {
  state.selectedPath = pathText || "";
  elements.sourcePath.value = state.selectedPath;
  elements.currentPathLabel.textContent = state.selectedPath || "尚未选择目录";
}

function normalizeMaxAbstractionsInput(value) {
  const raw = String(value ?? "").trim();
  return raw || "auto";
}

function renderTasks(snapshot) {
  elements.queueMode.textContent = snapshot.auto_run ? "运行中" : "待机";
  elements.activeTask.textContent = snapshot.active_task_id || "无";
  elements.pendingCount.textContent = `${snapshot.pending_count} 待执行`;
  elements.completedCount.textContent = `${snapshot.completed_count} 已完成`;
  elements.failedCount.textContent = `${snapshot.failed_count} 失败`;

  const tasks = snapshot.tasks || [];
  if (tasks.length === 0) {
    elements.taskList.innerHTML = '<div class="empty-state">队列还没有任务，先从左侧选择目录并加入一个任务。</div>';
    state._taskCards = {};
    return;
  }

  const existing = state._taskCards || {};
  const seen = {};

  // 移除已不存在的任务卡片
  Object.keys(existing).forEach((id) => {
    if (!tasks.some((t) => t.id === id)) {
      existing[id].remove();
      delete existing[id];
    }
  });

  tasks.slice().reverse().forEach((task) => {
    const id = task.id;
    seen[id] = true;
    let card = existing[id];

    if (!card) {
      // 新任务 → 创建卡片
      card = document.createElement("article");
      card.className = "task-card";
      card.dataset.taskId = id;
      card.dataset.status = task.status;

      card.innerHTML = `
        <div class="task-card-header">
          <div>
            <strong class="task-title"></strong>
            <div class="task-path"></div>
          </div>
          <div class="task-actions">
            <span class="task-status"></span>
            <button type="button" class="ghost-button delete-task-button">删除任务</button>
          </div>
        </div>
        <div class="task-meta"></div>
        <div class="progress-wrap" style="display:none">
          <div class="progress-bar"><div class="progress-fill"></div></div>
          <span class="progress-label"></span>
        </div>
        <div class="task-result" style="display:none"></div>
        <pre class="task-log"></pre>
      `;

      const deleteButton = card.querySelector(".delete-task-button");
      deleteButton.addEventListener("click", () => deleteTask(id));
      existing[id] = card;
      elements.taskList.appendChild(card);
    }

    // ── 只更新内容，不动 DOM 结构 ──
    card.dataset.status = task.status;
    card.querySelector(".task-title").textContent = task.config.project_name || task.config.source_path;
    card.querySelector(".task-path").textContent = task.config.source_path;
    const statusEl = card.querySelector(".task-status");
    statusEl.className = `task-status status-${task.status}`;
    statusEl.textContent = task.status;
    card.querySelector(".delete-task-button").disabled = task.status === "running";

    const meta = `创建: ${task.created_at}${task.started_at ? ` | 开始: ${task.started_at}` : ""}${task.finished_at ? ` | 结束: ${task.finished_at}` : ""}`;
    card.querySelector(".task-meta").textContent = meta;

    const prog = task.progress || { current: 0, total: 6, name: "", percent: 0 };
    const pw = card.querySelector(".progress-wrap");
    if (prog.total > 0) {
      pw.style.display = "flex";
      pw.querySelector(".progress-fill").style.width = `${prog.percent}%`;
      pw.querySelector(".progress-label").textContent = `${prog.current}/${prog.total} ${prog.name}`;
    } else {
      pw.style.display = "none";
    }

    const resultEl = card.querySelector(".task-result");
    if (task.result_dir) {
      resultEl.style.display = "block";
      resultEl.textContent = `输出目录: ${task.result_dir}`;
      resultEl.style.color = "";
    } else if (task.error) {
      resultEl.style.display = "block";
      resultEl.textContent = `错误: ${task.error}`;
      resultEl.style.color = "var(--danger)";
    } else {
      resultEl.style.display = "none";
    }

    // 日志：只更新 textContent，不替换 <pre> 元素自身
    const pre = card.querySelector(".task-log");
    const newLog = (task.logs || []).join("\n") || "暂无日志";
    if (pre.textContent !== newLog) {
      pre.textContent = newLog;
    }
  });

  state._taskCards = existing;
}

async function loadDefaults() {
  const defaults = await request("/api/defaults");
  state.defaults = defaults;
  elements.outputDir.value = defaults.output_dir;
  elements.language.value = defaults.language;
  elements.maxFileSize.value = defaults.max_file_size;
  elements.maxAbstractions.value = normalizeMaxAbstractionsInput(defaults.max_abstraction_num);
  elements.includePatterns.value = defaults.include_patterns;
  elements.excludePatterns.value = defaults.exclude_patterns;
}

async function pickFolder() {
  try {
    const payload = await request("/api/pick-folder", {
      method: "POST",
      body: "{}",
    });
    setSelectedPath(payload.selected_path);
    elements.outputDir.value = payload.output_dir;
    setMessage("已选择分析目录。");
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function refreshState() {
  try {
    const snapshot = await request("/api/state");
    renderTasks(snapshot);
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function deleteTask(taskId) {
  try {
    await request(`/api/tasks/${taskId}`, {
      method: "DELETE",
    });
    setMessage("任务已删除。");
    await refreshState();
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function addTask(event) {
  event.preventDefault();
  setMessage("");

  const payload = {
    source_path: elements.sourcePath.value,
    project_name: elements.projectName.value,
    output_dir: elements.outputDir.value,
    language: elements.language.value,
    use_cache: elements.useCache.checked,
    max_file_size: elements.maxFileSize.value,
    max_abstraction_num: normalizeMaxAbstractionsInput(elements.maxAbstractions.value),
    max_extraction_batches: elements.maxBatches.value,
    llm_extraction_concurrency: elements.llmConcurrency.value,
    include_patterns: elements.includePatterns.value,
    exclude_patterns: elements.excludePatterns.value,
  };

  try {
    await request("/api/tasks", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    setMessage("任务已加入队列。");
    await refreshState();
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function startQueue() {
  try {
    await request("/api/start", {
      method: "POST",
      body: "{}",
    });
    setMessage("任务队列已启动。");
    await refreshState();
  } catch (error) {
    setMessage(error.message, true);
  }
}

function bindEvents() {
  elements.taskForm.addEventListener("submit", addTask);
  elements.startQueueButton.addEventListener("click", startQueue);
  elements.pickFolderButton.addEventListener("click", pickFolder);
}

async function init() {
  bindEvents();
  await loadDefaults();
  await refreshState();
  state.pollHandle = window.setInterval(refreshState, 2500);
}

init();
