"""Animated running-viking SVG shown while RAGnar is thinking."""

_VIKING_RUNNING_HTML = """\
<div style="display:flex;align-items:center;gap:12px;padding:8px 0;">
<style>
@keyframes vk-leg-back { 0%,100% { transform: rotate(28deg); } 50% { transform: rotate(-28deg); } }
@keyframes vk-leg-front { 0%,100% { transform: rotate(-28deg); } 50% { transform: rotate(28deg); } }
@keyframes vk-arm-back { 0%,100% { transform: rotate(-25deg); } 50% { transform: rotate(25deg); } }
@keyframes vk-arm-front { 0%,100% { transform: rotate(25deg); } 50% { transform: rotate(-25deg); } }
@keyframes vk-bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-3px); } }
@keyframes vk-dust { 0% { opacity:0.7; transform:translateX(0); } 100% { opacity:0; transform:translateX(-12px); } }
@keyframes vk-dust2 { 0% { opacity:0.5; transform:translateX(0); } 100% { opacity:0; transform:translateX(-8px); } }
.vk-leg-back { animation: vk-leg-back .4s infinite ease-in-out; transform-box: fill-box; transform-origin: top center; }
.vk-leg-front { animation: vk-leg-front .4s infinite ease-in-out; transform-box: fill-box; transform-origin: top center; }
.vk-arm-back { animation: vk-arm-back .4s infinite ease-in-out; transform-box: fill-box; transform-origin: top center; }
.vk-arm-front { animation: vk-arm-front .4s infinite ease-in-out; transform-box: fill-box; transform-origin: top center; }
.vk-body { animation: vk-bob .4s infinite ease-in-out; }
.vk-dust { animation: vk-dust .5s infinite ease-out; }
.vk-dust2 { animation: vk-dust2 .5s infinite ease-out .15s; }
</style>
<svg viewBox="0 0 90 80" width="64" height="56">
  <!-- Ground line -->
  <line x1="5" y1="68" x2="75" y2="68" stroke="#555" stroke-width="1" opacity="0.3"/>
  <!-- Dust -->
  <g fill="#aaa">
    <circle class="vk-dust" cx="15" cy="66" r="2.5"/>
    <circle class="vk-dust2" cx="10" cy="64" r="1.8"/>
  </g>
  <g class="vk-body">
  <!-- Shield -->
  <g transform="translate(16,38)">
    <circle cx="0" cy="0" r="10" fill="#8B4513" stroke="#5C3317" stroke-width="2"/>
    <circle cx="0" cy="0" r="4" fill="#C0C0C0" stroke="#888" stroke-width="0.8"/>
    <line x1="-8" y1="0" x2="8" y2="0" stroke="#5C3317" stroke-width="0.8"/>
    <line x1="0" y1="-8" x2="0" y2="8" stroke="#5C3317" stroke-width="0.8"/>
  </g>
  <!-- Body / tunic -->
  <rect x="36" y="30" width="16" height="20" rx="3" fill="#3B6B8A"/>
  <!-- Belt buckle -->
  <rect x="36" y="44" width="16" height="3" fill="#5C3317"/>
  <rect x="42" y="43" width="4" height="5" fill="#C0C0C0"/>
  <!-- Head -->
  <circle cx="44" cy="22" r="9" fill="#FDBCB4"/>
  <!-- Eyes -->
  <circle cx="41" cy="21" r="1" fill="#333"/>
  <circle cx="47" cy="21" r="1" fill="#333"/>
  <!-- Beard -->
  <path d="M37 25 Q44 35 51 25 L48 29 Q44 32 40 29 Z" fill="#9B8B6F"/>
  <!-- Mustache -->
  <path d="M40 24 Q42 26 44 24 Q46 26 48 24" stroke="#9B8B6F" stroke-width="1.5" fill="none"/>
  <!-- Helmet -->
  <path d="M35 21 Q44 8 53 21 L50 24 L38 24 Z" fill="#777" stroke="#555" stroke-width="1"/>
  <!-- Helmet nose guard -->
  <rect x="43" y="19" width="2" height="5" fill="#555"/>
  <!-- Left horn -->
  <path d="M35 21 Q28 16 25 9 L31 20 Z" fill="#F0E6D2" stroke="#C8B898" stroke-width="0.8"/>
  <!-- Right horn -->
  <path d="M53 21 Q60 16 63 9 L57 20 Z" fill="#F0E6D2" stroke="#C8B898" stroke-width="0.8"/>
  <!-- Back arm (holds shield) -->
  <g class="vk-arm-back">
    <rect x="32" y="32" width="5" height="14" rx="2.5" fill="#3B6B8A"/>
    <circle cx="34.5" cy="46" r="2.5" fill="#FDBCB4"/>
  </g>
  <!-- Front arm -->
  <g class="vk-arm-front">
    <rect x="50" y="32" width="5" height="14" rx="2.5" fill="#3B6B8A"/>
    <circle cx="52.5" cy="46" r="2.5" fill="#FDBCB4"/>
  </g>
  <!-- Back leg -->
  <g class="vk-leg-back">
    <rect x="38" y="48" width="5.5" height="16" rx="2.75" fill="#2C3E50"/>
    <ellipse cx="40" cy="65" rx="4.5" ry="2.5" fill="#1a1a1a"/>
  </g>
  <!-- Front leg -->
  <g class="vk-leg-front">
    <rect x="45" y="48" width="5.5" height="16" rx="2.75" fill="#2C3E50"/>
    <ellipse cx="47" cy="65" rx="4.5" ry="2.5" fill="#1a1a1a"/>
  </g>
  </g>
</svg>
<div style="display:flex;flex-direction:column;gap:2px;">
  <span style="color:#8B4513;font-size:0.9rem;font-weight:700;">RAGnar is running…</span>
  <span style="color:#aaa;font-size:0.75rem;">Searching through your documents</span>
</div>
</div>
"""


def viking_running_html() -> str:
    """Return inline HTML for the running-viking thinking indicator."""
    return _VIKING_RUNNING_HTML