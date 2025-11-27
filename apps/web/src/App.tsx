import { useState } from "react";
import "./App.css";

export default function App() {
  const [prompts, setPrompts] = useState<string[]>([]);
  const [draft, setDraft] = useState("");

  const handleAdd = () => {
    if (!draft.trim()) return;
    setPrompts((items) => [draft.trim(), ...items]);
    setDraft("");
  };

  return (
    <main className="web-app">
      <section>
        <h1>PromptLab Web</h1>
        <p>浏览器端样板，稍后可与后端 API 对接。</p>
        <div className="composer">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="快速记录一个 prompt..."
          />
          <button type="button" onClick={handleAdd}>保存</button>
        </div>
      </section>

      <section>
        <h2>最近记录</h2>
        <ul>
          {prompts.length === 0 && <li>暂无数据</li>}
          {prompts.map((item, index) => (
            <li key={`${item}-${index}`}>{item}</li>
          ))}
        </ul>
      </section>
    </main>
  );
}
