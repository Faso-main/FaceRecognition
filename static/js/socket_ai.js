var socket = new WebSocket('ws://' + window.location.host + '/tasks/')
var tasks = document.getElementsByClassName('tasks')[0]
var autorization_box = document.getElementsByClassName('ai-autoriz')[0]
var user = document.getElementsByClassName('user-is')[0]
var user_name = user.getElementsByTagName('h4')[0]
var tid = 0

socket.onmessage = function(event) {
    let obj = JSON.parse(event.data)
    let now_user = `${obj['prepodavatel']['surname']} ${obj['prepodavatel']['name']} ${obj['prepodavatel']['secondname']}`
    if (now_user == user_name.textContent) {
      clearTimeout(tid)
      tid = setTimeout(clearAllInfo, 10000)
    }
    if(now_user != user_name.textContent) {
      clearTasks()
      clearTimeout(tid)
      autorization_box.style.display = 'none'
      user_name.textContent = now_user
      user.getElementsByTagName('p')[0].textContent = `Должность: ${obj['prepodavatel']['post']}`
      for(let i in obj['tasks']){
        let div = document.createElement('div')
        div.setAttribute('class', 'task')
        div.innerHTML = `
          <h4>ID: ${obj['tasks'][i]['id']}</h4>
          <p>${obj['tasks'][i]['description']}</p>
          <h6>${obj['tasks'][i]['date_end']}</h6>
        `
        tasks.appendChild(div)
      }
      tid = setTimeout(clearAllInfo, 10000)
    }
    
}

function clearTasks() {
  while(tasks.firstChild){
    tasks.removeChild(tasks.firstChild)
  }
}

function clearAllInfo() {
  user_name.textContent = ''
  user.getElementsByTagName('p')[0].textContent = ''
  clearTasks()
  autorization_box.style.display = 'flex'
}