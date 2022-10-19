const id = JSON.parse(document.getElementById('json-username').textContent);
const message_username = JSON.parse(document.getElementById('json-message-username').textContent);
const userclass = JSON.parse(document.getElementById('json-userclass').textContent);
const inference_result = JSON.parse(document.getElementById('json-inference-result').textContent);

const socket = new WebSocket(
    'ws://'
    + window.location.host
    + '/ws/'
    + id
    + '/'
);

socket.onopen = function(e){
    console.log("CONNECTION ESTABLISHED");
}

socket.onclose = function(e){
    console.log("CONNECTION LOST");
}

socket.onerror = function(e){
    console.log("ERROR OCCURED");
}

socket.onmessage = function(e){
    const data = JSON.parse(e.data);
    console.log(data)
    
    if (data.vid_dir){
        if(data.username == message_username){
            document.querySelector('.chatroom').innerHTML += `<div data-sent="true" class="chat_area">
                                                                <img src="/static/assets/dp.png" alt="" class="chat_profile" />
                                                                <div class="chatroom_chat">
                                                                <span class="receive_message_sender">${data.username}</span>
                                                                <div class="chat_content">
                                                                    <div class="chat_message">${data.message}<br /></div>
                                                                </div>
                                                                <video width="252px" height="141px" class="shadow-sm float-left" style="border-radius:15px;" controls>
                                                                    <source src="/static/${data.vid_dir}" type="video/mp4"> </source> 
                                                                </video>            
                                                            </div>`
        }else{
            document.querySelector('.chatroom').innerHTML += `<div data-sent="false" class="chat_area">
                                                                <img src="/static/assets/dp.png" alt="" class="chat_profile" />
                                                                <div class="chatroom_chat">
                                                                <span class="receive_message_sender">${data.username}</span>
                                                                <div class="chat_content">
                                                                    <div class="chat_message">${data.message}<br /></div>
                                                                </div>
                                                                <video width="252px" height="141px" class="shadow-sm float-left" style="border-radius:15px;" controls>
                                                                    <source src="/static/${data.vid_dir}" type="video/mp4"> </source> 
                                                                </video>
                                                            </div>`
        }

    } else {
        if(data.username == message_username){
            document.querySelector('.chatroom').innerHTML += `<div data-sent="true" class="chat_area">
                                                                <img src="/static/assets/dp.png" alt="" class="chat_profile" />
                                                                <div class="chatroom_chat">
                                                                <span class="receive_message_sender">${data.username}</span>
                                                                <div class="chat_content">
                                                                    <div class="chat_message">${data.message}<br /></div>
                                                                </div>
                                                            </div>`
        }else{
            document.querySelector('.chatroom').innerHTML += `<div data-sent="false" class="chat_area">
                                                                <img src="/static/assets/dp.png" alt="" class="chat_profile" />
                                                                <div class="chatroom_chat">
                                                                <span class="receive_message_sender">${data.username}</span>
                                                                <div class="chat_content">
                                                                    <div class="chat_message">${data.message}<br /></div>
                                                                </div>
                                                            </div>`
        }
    }
    
    document.getElementById("scroll").scrollTop = document.getElementById("scroll").scrollHeight;    
}

document.querySelector('#chat-message-submit').onclick = function(e){
    const message_input = document.querySelector('#message_input');
    const message = message_input.value;

    console.log(message)

    socket.send(JSON.stringify({
        'message':message,
        'username':message_username
    }));

    message_input.value = '';
}

document.addEventListener("DOMContentLoaded", () => {
    console.log(document.getElementById("scroll").scrollHeight)
    document.getElementById("scroll").scrollTop =
      document.getElementById("scroll").scrollHeight;
  });

  document.querySelector('#message_input').addEventListener('keydown', (event) => {
    if(e.code == 'Enter') {
        const message_input = document.querySelector("#message_input");
        const message = message_input.value;

        console.log(message)

        socket.send(
            JSON.stringify({
                message: message,
                username: message_username,
            })
        )
    }
});

    document.querySelector("form").onsubmit = function (e) {
        const message_input = document.querySelector("#message_input");
        const message = message_input.value;

        console.log(message)

        socket.send(
            JSON.stringify({
                message: message,
                username: message_username,
        })
    );

    message_input.value = "";
    };