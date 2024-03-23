window.addEventListener("load", () => {
const about = document.getElementById("about");
document.addEventListener("keyup" ,event => {
    about.style.backgroundColor = "grey"
});
document.addEventListener("keydown" ,event => {
    about.style.backgroundColor = "red"
});
});
document.getElementById("about").addEventListener('click', function() {
    window.open('https://google.com', '_blank');
});
document.getElementById('about').addEventListener('click', function() {
    window.open('https://google.com', '_blank');
});
