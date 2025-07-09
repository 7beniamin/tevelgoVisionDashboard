function toggleSidebar() {
  const sidebar = document.getElementById("mySidebar");
  if (!sidebar) return;
  sidebar.style.left = (sidebar.style.left === "0px") ? "-250px" : "0";
}

// Attach event listener to sidebar toggle button
document.addEventListener("DOMContentLoaded", function() {
  const sidebarToggle = document.getElementById("sidebarToggle");
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", function(e){
      e.stopPropagation();
      toggleSidebar();
    });
  }

  // Close sidebar when clicking outside of it
  document.addEventListener("click", function(event) {
    const sidebar = document.getElementById("mySidebar");
    const toggleBtn = document.getElementById("sidebarToggle");
    if (
      sidebar &&
      sidebar.style.left === "0px" &&
      !sidebar.contains(event.target) &&
      event.target !== toggleBtn
    ) {
      sidebar.style.left = "-250px";
    }
  });
});
