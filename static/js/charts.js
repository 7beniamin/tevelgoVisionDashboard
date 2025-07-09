document.addEventListener("DOMContentLoaded", function() {
  const el = document.getElementById('summaryChart');
  if (!el) return;
  const getDataAttr = (name) => {
    // Try to read attribute, parseInt, default to 0 if missing, empty, null, or NaN
    const val = el.getAttribute('data-' + name);
    const parsed = parseInt(val, 10);
    return isNaN(parsed) ? 0 : parsed;
  };
  const ctx = el.getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [
        'Vehicle Success',
        'Vehicle Fail',
        'Vehicle Mismatch',
        'Pedestrian Success',
        'Pedestrian Fail'
      ],
      datasets: [{
        label: 'Attempts',
        data: [
          getDataAttr('vehicle-success'),
          getDataAttr('vehicle-fail'),
          getDataAttr('vehicle-mismatch'),
          getDataAttr('pedestrian-success'),
          getDataAttr('pedestrian-fail')
        ],
        backgroundColor: [
          '#00B894','#FF4D4D','#FFD166','#00BFFF','#FF4D4D'
        ],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  });
});
