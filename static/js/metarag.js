addEventListener("DOMContentLoaded", main)

function main() {
    document.getElementById("user-input-form").addEventListener("submit", inputSubmitted);
}

function inputSubmitted(e){
    e.preventDefault()
    let templates = document.getElementById("chat-templates")
    let userBox = templates.content.querySelector(".user-box")

    let userInput = document.getElementById("user-input")
    userBox.querySelector(".utterance").innerHTML = userInput.value
    userInput.value = "";

    document.getElementById("chat-window").appendChild(userBox)
}