// Utility to safely parse ISO date string (handles Azure style and Zulu/UTC)
function safeDate(iso) {
  if (!iso) return null;
  let s = iso.replace('Z', '');
  if (s.includes('.')) s = s.split('.')[0];
  let dt = new Date(s);
  return isNaN(dt.getTime()) ? null : dt;
}

function localizeLogDates() {
  // Set user-local date
  document.querySelectorAll('.user-local-date').forEach(function(cell) {
    const iso = cell.getAttribute('data-timestamp');
    const dt = safeDate(iso);
    if (dt) {
      const date = (dt.getMonth() + 1) + '/' + dt.getDate() + '/' + dt.getFullYear();
      cell.textContent = date;
    } else {
      cell.textContent = "N/A";
    }
  });

  // Set user-local time
  document.querySelectorAll('.user-local-time').forEach(function(cell) {
    const iso = cell.getAttribute('data-timestamp');
    const dt = safeDate(iso);
    if (dt) {
      let hours = dt.getHours();
      let minutes = dt.getMinutes().toString().padStart(2, '0');
      const ampm = hours >= 12 ? 'pm' : 'am';
      hours = hours % 12;
      hours = hours ? hours : 12;
      const time = hours + ':' + minutes + ' ' + ampm;
      cell.textContent = time;
    } else {
      cell.textContent = "N/A";
    }
  });
}

// On DOM ready
document.addEventListener("DOMContentLoaded", localizeLogDates);
