function renderExplanationInChat(explanationText) {
  const log = document.getElementById("chatLog");

  log.innerHTML = `
    <div class="chat-msg assistant">
      ${marked.parse(explanationText)}
    </div>
  `;

  log.scrollTop = log.scrollHeight;
}