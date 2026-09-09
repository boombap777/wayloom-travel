"use strict";
const $ = (id) => document.getElementById(id);
let context = [],
  latest = null,
  busy = false;
const labels = {
  departure_city: "出发城市",
  destination: "目的地",
  start_date: "出发日期",
  days: "旅行天数",
  traveler_count: "出行人数",
  budget_cny: "总预算",
  themes: "旅行偏好",
  hotel_preference: "酒店偏好",
};
const examples = {
  full: "从南京去杭州，10月1日出发玩三天，两个人，预算5000元，喜欢人文和美食。",
  short: "我想去杭州旅游。",
};
function node(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
function status(text, kind = "") {
  $("status").textContent = text;
  $("status").dataset.state = kind;
}
function setBusy(value) {
  busy = value;
  for (const id of [
    "submit",
    "reset",
    "example-full",
    "example-short",
    "adjust",
  ])
    $(id).disabled = value;
  $("request").disabled = value;
  $("submit").textContent = value ? "正在解析需求…" : "提交需求 →";
  $("request-form").setAttribute("aria-busy", String(value));
}
function step(index) {
  for (let i = 1; i <= 3; i++) {
    if (i === index) $("step-" + i).setAttribute("aria-current", "step");
    else $("step-" + i).removeAttribute("aria-current");
  }
}
function bubble(who, text) {
  const el = node("div", undefined, "bubble " + who);
  el.append(
    node("strong", who === "user" ? "你" : "旅行助手"),
    node("span", text),
  );
  $("conversation").append(el);
}
function reset(text = "") {
  if (busy) return;
  context = [];
  latest = null;
  $("conversation").replaceChildren();
  $("request-summary").replaceChildren();
  $("summary-empty").hidden = false;
  $("missing").hidden = true;
  $("result").hidden = true;
  $("request").value = text;
  $("request").disabled = false;
  $("submit").disabled = false;
  $("request-label").textContent = "描述你的旅行需求";
  step(1);
  status("已开始新的需求，不会携带上一次的信息。");
  $("request").focus();
}
function show(data) {
  const response = data.response;
  latest = data;
  $("request-summary").replaceChildren();
  $("summary-empty").hidden = true;
  for (const [key, label] of Object.entries(labels)) {
    let value = response.request[key];
    if (Array.isArray(value)) value = value.join("、");
    $("request-summary").append(
      node("dt", label),
      node("dd", value === null || value === "" ? "未提供" : String(value)),
    );
  }
  bubble("assistant", response.message);
  if (response.action === "clarify") {
    step(2);
    $("missing").hidden = false;
    $("missing").textContent =
      "还需要：" +
      response.decision.missing_fields.map((x) => labels[x] || x).join("、");
    $("request-label").textContent = "补充缺少的信息";
    $("result").hidden = true;
    status("已保留你提供的信息，请继续补充。", "success");
    $("request").focus();
    return;
  }
  step(3);
  $("missing").hidden = true;
  $("result").hidden = false;
  $("result-message").textContent = response.message;
  $("plan-content").replaceChildren();
  $("json-result").textContent = JSON.stringify(data, null, 2);
  const plan = response.plan;
  if (plan.selected_option) {
    const grid = node("div", undefined, "plan-grid");
    const info = node("div");
    info.append(
      node("h3", plan.selected_option.title, "plan-title"),
      node("p", "样例预算 · 非实时报价", "muted"),
      node("div", "¥ " + plan.estimated_total_cny.toLocaleString(), "budget"),
      node("p", plan.disclaimer, "muted"),
    );
    const activities = node("div");
    activities.append(node("p", "目录参考活动 · 并非按天生成的排程", "muted"));
    const list = node("ol", undefined, "outline");
    plan.selected_option.outline.forEach((text, i) =>
      list.append(node("li", String(i + 1).padStart(2, "0") + "  " + text)),
    );
    activities.append(list);
    grid.append(info, activities);
    $("plan-content").append(grid);
    $("result-heading").textContent = "你的旅行草案";
  } else {
    $("result-heading").textContent = "暂未找到合适的目录样例";
    $("plan-content").append(
      node(
        "p",
        "演示目录包含杭州、成都、厦门、北京和西安。你可以调整需求后重新查询。",
        "muted",
      ),
    );
  }
  $("request").disabled = true;
  $("submit").disabled = true;
  status(
    "本次处理完成 · " + (data.elapsed_ms < 1 ? "< 1" : Math.round(data.elapsed_ms)) + " ms · 未执行任何预订",
    "success",
  );
  $("result").focus();
}
$("request-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const message = $("request").value.trim();
  if (!message) {
    status("请先输入需求。", "error");
    return;
  }
  setBusy(true);
  status("正在等待本地解析器返回经过校验的结果。");
  try {
    const res = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, context }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "服务暂不可用，请重试。");
    context.push(message);
    bubble("user", message);
    $("request").value = "";
    setBusy(false);
    show(data);
  } catch (error) {
    setBusy(false);
    status(error.message || "连接失败，请检查本地服务。", "error");
  }
});
$("example-full").addEventListener("click", () => reset(examples.full));
$("example-short").addEventListener("click", () => reset(examples.short));
$("reset").addEventListener("click", () => reset());
$("adjust").addEventListener("click", () => reset(context.join("。补充：")));
$("download").addEventListener("click", () => {
  if (!latest) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(latest, null, 2)], { type: "application/json" }),
  );
  const link = node("a");
  link.href = url;
  link.download = "travel-draft.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
fetch("/api/status")
  .then((res) => {
    if (!res.ok) throw new Error();
    return res.json();
  })
  .then((data) => {
    const names = {
      rule: "规则演示 · 不使用大模型",
      hf_adapter: "HF adapter · 本地模型",
      llama_cpp: "llama.cpp · 本地模型",
    };
    $("mode").textContent = names[data.profile] || data.profile;
    $("mode-detail").textContent =
      "服务器选定模式：" +
      data.profile +
      "；日期基准：" +
      data.today +
      "。模型模式首次请求时加载或验证本地制品，失败不会回退成规则结果。";
  })
  .catch(() => {
    status("连接失败，请确认本地服务已启动。", "error");
    $("mode").textContent = "服务暂不可用";
  });
