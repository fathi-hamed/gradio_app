import { c as create_ssr_component, a as createEventDispatcher, v as validate_component, s as subscribe, p as setContext, q as set_store_value, o as onDestroy, e as escape, b as add_attribute, f as each } from './ssr-RaXq3SJh.js';
import { w as writable } from './index-hSrgoQUm.js';

const OverflowIcon = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  return `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="2.5" cy="8" r="1.5" fill="currentColor"></circle><circle cx="8" cy="8" r="1.5" fill="currentColor"></circle><circle cx="13.5" cy="8" r="1.5" fill="currentColor"></circle></svg>`;
});
const css = {
  code: '.tabs.svelte-r9stgs.svelte-r9stgs{position:relative}.hide.svelte-r9stgs.svelte-r9stgs{display:none}.tab-wrapper.svelte-r9stgs.svelte-r9stgs{display:flex;align-items:center;justify-content:space-between;position:relative;height:var(--size-8);padding-bottom:var(--size-2)}.tab-container.svelte-r9stgs.svelte-r9stgs{display:flex;align-items:center;width:100%;position:relative;overflow:hidden;height:var(--size-8)}.tab-container.svelte-r9stgs.svelte-r9stgs::after{content:"";position:absolute;bottom:0;left:0;right:0;height:1px;background-color:var(--border-color-primary)}.overflow-menu.svelte-r9stgs.svelte-r9stgs{flex-shrink:0;margin-left:var(--size-2)}button.svelte-r9stgs.svelte-r9stgs{margin-bottom:0;border:none;border-radius:0;padding:0 var(--size-4);color:var(--body-text-color);font-weight:var(--section-header-text-weight);font-size:var(--section-header-text-size);transition:all 0.2s ease-out;background-color:transparent;height:100%;display:flex;align-items:center;white-space:nowrap;position:relative}button.svelte-r9stgs.svelte-r9stgs:disabled{opacity:0.5;cursor:not-allowed}button.svelte-r9stgs.svelte-r9stgs:hover:not(:disabled):not(.selected){background-color:var(--background-fill-secondary);color:var(--body-text-color)}.selected.svelte-r9stgs.svelte-r9stgs{background-color:transparent;color:var(--color-accent);position:relative}.selected.svelte-r9stgs.svelte-r9stgs::after{content:"";position:absolute;bottom:0;left:0;width:100%;height:2px;background-color:var(--color-accent);animation:svelte-r9stgs-fade-grow 0.2s ease-out forwards;transform-origin:center;z-index:1}@keyframes svelte-r9stgs-fade-grow{from{opacity:0;transform:scaleX(0.8)}to{opacity:1;transform:scaleX(1)}}.overflow-dropdown.svelte-r9stgs.svelte-r9stgs{position:absolute;top:calc(100% + var(--size-2));right:0;background-color:var(--background-fill-primary);border:1px solid var(--border-color-primary);border-radius:var(--radius-sm);z-index:var(--layer-5);box-shadow:0 2px 5px rgba(0, 0, 0, 0.1);padding:var(--size-2);min-width:150px;width:max-content}.overflow-dropdown.svelte-r9stgs button.svelte-r9stgs{display:block;width:100%;text-align:left;padding:var(--size-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.overflow-menu.svelte-r9stgs>button.svelte-r9stgs{padding:var(--size-1) var(--size-2);min-width:auto;border:1px solid var(--border-color-primary);border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center}.overflow-menu.svelte-r9stgs>button.svelte-r9stgs:hover{background-color:var(--background-fill-secondary)}.overflow-menu.svelte-r9stgs svg{width:16px;height:16px}.overflow-item-selected.svelte-r9stgs svg{color:var(--color-accent)}',
  map: '{"version":3,"file":"Tabs.svelte","sources":["Tabs.svelte"],"sourcesContent":["<script context=\\"module\\">\\n\\texport const TABS = {};\\n<\/script>\\n\\n<script lang=\\"ts\\">import { setContext, createEventDispatcher, onMount, onDestroy } from \\"svelte\\";\\nimport OverflowIcon from \\"./OverflowIcon.svelte\\";\\nimport { writable } from \\"svelte/store\\";\\nexport let visible = true;\\nexport let elem_id = \\"\\";\\nexport let elem_classes = [];\\nexport let selected;\\nlet tabs = [];\\nlet overflow_menu_open = false;\\nlet overflow_menu;\\n$: has_tabs = tabs.length > 0;\\nlet tab_nav_el;\\nlet overflow_nav;\\nconst selected_tab = writable(false);\\nconst selected_tab_index = writable(0);\\nconst dispatch = createEventDispatcher();\\nlet is_overflowing = false;\\nlet overflow_has_selected_tab = false;\\nsetContext(TABS, {\\n    register_tab: (tab) => {\\n        let index = tabs.findIndex((t) => t.id === tab.id);\\n        if (index !== -1) {\\n            tabs[index] = { ...tabs[index], ...tab };\\n        }\\n        else {\\n            tabs = [...tabs, tab];\\n            index = tabs.length - 1;\\n        }\\n        if ($selected_tab === false && tab.visible && tab.interactive) {\\n            $selected_tab = tab.id;\\n        }\\n        return index;\\n    },\\n    unregister_tab: (tab) => {\\n        const index = tabs.findIndex((t) => t.id === tab.id);\\n        if (index !== -1) {\\n            tabs = tabs.filter((t) => t.id !== tab.id);\\n            if ($selected_tab === tab.id) {\\n                $selected_tab = tabs[0]?.id || false;\\n            }\\n        }\\n    },\\n    selected_tab,\\n    selected_tab_index\\n});\\nfunction change_tab(id) {\\n    const tab_to_activate = tabs.find((t) => t.id === id);\\n    if (tab_to_activate && tab_to_activate.interactive && tab_to_activate.visible) {\\n        selected = id;\\n        $selected_tab = id;\\n        $selected_tab_index = tabs.findIndex((t) => t.id === id);\\n        dispatch(\\"change\\");\\n        overflow_menu_open = false;\\n    }\\n    else {\\n        console.warn(\\"Attempted to select a non-interactive or hidden tab.\\");\\n    }\\n}\\n$: tabs, selected !== null && change_tab(selected);\\nonMount(() => {\\n    handle_menu_overflow();\\n    window.addEventListener(\\"resize\\", handle_menu_overflow);\\n    window.addEventListener(\\"click\\", handle_outside_click);\\n});\\nonDestroy(() => {\\n    if (typeof window === \\"undefined\\")\\n        return;\\n    window.removeEventListener(\\"resize\\", handle_menu_overflow);\\n    window.removeEventListener(\\"click\\", handle_outside_click);\\n});\\nfunction handle_outside_click(event) {\\n    if (overflow_menu_open && overflow_menu && !overflow_menu.contains(event.target)) {\\n        overflow_menu_open = false;\\n    }\\n}\\nfunction handle_menu_overflow() {\\n    if (!tab_nav_el) {\\n        console.error(\\"Menu elements not found\\");\\n        return;\\n    }\\n    let all_items = [];\\n    [tab_nav_el, overflow_nav].forEach((menu) => {\\n        Array.from(menu.querySelectorAll(\\"button\\")).forEach((item) => all_items.push(item));\\n    });\\n    all_items.forEach((item) => tab_nav_el.appendChild(item));\\n    const nav_items = [];\\n    const overflow_items = [];\\n    Array.from(tab_nav_el.querySelectorAll(\\"button\\")).forEach((item) => {\\n        const tab_rect = item.getBoundingClientRect();\\n        const tab_menu_rect = tab_nav_el.getBoundingClientRect();\\n        is_overflowing = tab_rect.right > tab_menu_rect.right || tab_rect.left < tab_menu_rect.left;\\n        if (is_overflowing) {\\n            overflow_items.push(item);\\n        }\\n        else {\\n            nav_items.push(item);\\n        }\\n    });\\n    nav_items.forEach((item) => tab_nav_el.appendChild(item));\\n    overflow_items.forEach((item) => overflow_nav.appendChild(item));\\n    overflow_has_selected_tab = tabs.some((t) => t.id === $selected_tab && overflow_nav.contains(document.querySelector(`[data-tab-id=\\"${t.id}\\"]`)));\\n}\\n<\/script>\\n\\n{#if has_tabs}\\n\\t<div class=\\"tabs {elem_classes.join(\' \')}\\" class:hide={!visible} id={elem_id}>\\n\\t\\t<div class=\\"tab-wrapper\\">\\n\\t\\t\\t<div class=\\"tab-container\\" bind:this={tab_nav_el} role=\\"tablist\\">\\n\\t\\t\\t\\t{#each tabs as t, i (t.id)}\\n\\t\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\t\\trole=\\"tab\\"\\n\\t\\t\\t\\t\\t\\tclass:selected={t.id === $selected_tab}\\n\\t\\t\\t\\t\\t\\taria-selected={t.id === $selected_tab}\\n\\t\\t\\t\\t\\t\\taria-controls={t.elem_id}\\n\\t\\t\\t\\t\\t\\tdisabled={!t.interactive}\\n\\t\\t\\t\\t\\t\\taria-disabled={!t.interactive}\\n\\t\\t\\t\\t\\t\\tid={t.elem_id ? t.elem_id + \\"-button\\" : null}\\n\\t\\t\\t\\t\\t\\tdata-tab-id={t.id}\\n\\t\\t\\t\\t\\t\\ton:click={() => {\\n\\t\\t\\t\\t\\t\\t\\tif (t.id !== $selected_tab) {\\n\\t\\t\\t\\t\\t\\t\\t\\tchange_tab(t.id);\\n\\t\\t\\t\\t\\t\\t\\t\\tdispatch(\\"select\\", { value: t.name, index: i });\\n\\t\\t\\t\\t\\t\\t\\t}\\n\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t{t.name}\\n\\t\\t\\t\\t\\t</button>\\n\\t\\t\\t\\t{/each}\\n\\t\\t\\t</div>\\n\\t\\t\\t<span\\n\\t\\t\\t\\tclass=\\"overflow-menu\\"\\n\\t\\t\\t\\tclass:hide={!is_overflowing}\\n\\t\\t\\t\\tbind:this={overflow_menu}\\n\\t\\t\\t>\\n\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\ton:click|stopPropagation={() =>\\n\\t\\t\\t\\t\\t\\t(overflow_menu_open = !overflow_menu_open)}\\n\\t\\t\\t\\t\\tclass:overflow-item-selected={overflow_has_selected_tab}\\n\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t<OverflowIcon />\\n\\t\\t\\t\\t</button>\\n\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\tclass=\\"overflow-dropdown\\"\\n\\t\\t\\t\\t\\tbind:this={overflow_nav}\\n\\t\\t\\t\\t\\tclass:hide={!overflow_menu_open}\\n\\t\\t\\t\\t/>\\n\\t\\t\\t</span>\\n\\t\\t</div>\\n\\t</div>\\n{/if}\\n\\n<slot />\\n\\n<style>\\n\\t.tabs {\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\t.hide {\\n\\t\\tdisplay: none;\\n\\t}\\n\\n\\t.tab-wrapper {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: space-between;\\n\\t\\tposition: relative;\\n\\t\\theight: var(--size-8);\\n\\t\\tpadding-bottom: var(--size-2);\\n\\t}\\n\\n\\t.tab-container {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\twidth: 100%;\\n\\t\\tposition: relative;\\n\\t\\toverflow: hidden;\\n\\t\\theight: var(--size-8);\\n\\t}\\n\\n\\t.tab-container::after {\\n\\t\\tcontent: \\"\\";\\n\\t\\tposition: absolute;\\n\\t\\tbottom: 0;\\n\\t\\tleft: 0;\\n\\t\\tright: 0;\\n\\t\\theight: 1px;\\n\\t\\tbackground-color: var(--border-color-primary);\\n\\t}\\n\\n\\t.overflow-menu {\\n\\t\\tflex-shrink: 0;\\n\\t\\tmargin-left: var(--size-2);\\n\\t}\\n\\n\\tbutton {\\n\\t\\tmargin-bottom: 0;\\n\\t\\tborder: none;\\n\\t\\tborder-radius: 0;\\n\\t\\tpadding: 0 var(--size-4);\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tfont-weight: var(--section-header-text-weight);\\n\\t\\tfont-size: var(--section-header-text-size);\\n\\t\\ttransition: all 0.2s ease-out;\\n\\t\\tbackground-color: transparent;\\n\\t\\theight: 100%;\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\twhite-space: nowrap;\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\tbutton:disabled {\\n\\t\\topacity: 0.5;\\n\\t\\tcursor: not-allowed;\\n\\t}\\n\\n\\tbutton:hover:not(:disabled):not(.selected) {\\n\\t\\tbackground-color: var(--background-fill-secondary);\\n\\t\\tcolor: var(--body-text-color);\\n\\t}\\n\\n\\t.selected {\\n\\t\\tbackground-color: transparent;\\n\\t\\tcolor: var(--color-accent);\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\t.selected::after {\\n\\t\\tcontent: \\"\\";\\n\\t\\tposition: absolute;\\n\\t\\tbottom: 0;\\n\\t\\tleft: 0;\\n\\t\\twidth: 100%;\\n\\t\\theight: 2px;\\n\\t\\tbackground-color: var(--color-accent);\\n\\t\\tanimation: fade-grow 0.2s ease-out forwards;\\n\\t\\ttransform-origin: center;\\n\\t\\tz-index: 1;\\n\\t}\\n\\n\\t@keyframes fade-grow {\\n\\t\\tfrom {\\n\\t\\t\\topacity: 0;\\n\\t\\t\\ttransform: scaleX(0.8);\\n\\t\\t}\\n\\t\\tto {\\n\\t\\t\\topacity: 1;\\n\\t\\t\\ttransform: scaleX(1);\\n\\t\\t}\\n\\t}\\n\\n\\t.overflow-dropdown {\\n\\t\\tposition: absolute;\\n\\t\\ttop: calc(100% + var(--size-2));\\n\\t\\tright: 0;\\n\\t\\tbackground-color: var(--background-fill-primary);\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tz-index: var(--layer-5);\\n\\t\\tbox-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);\\n\\t\\tpadding: var(--size-2);\\n\\t\\tmin-width: 150px;\\n\\t\\twidth: max-content;\\n\\t}\\n\\n\\t.overflow-dropdown button {\\n\\t\\tdisplay: block;\\n\\t\\twidth: 100%;\\n\\t\\ttext-align: left;\\n\\t\\tpadding: var(--size-2);\\n\\t\\twhite-space: nowrap;\\n\\t\\toverflow: hidden;\\n\\t\\ttext-overflow: ellipsis;\\n\\t}\\n\\n\\t.overflow-menu > button {\\n\\t\\tpadding: var(--size-1) var(--size-2);\\n\\t\\tmin-width: auto;\\n\\t\\tborder: 1px solid var(--border-color-primary);\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t}\\n\\n\\t.overflow-menu > button:hover {\\n\\t\\tbackground-color: var(--background-fill-secondary);\\n\\t}\\n\\n\\t.overflow-menu :global(svg) {\\n\\t\\twidth: 16px;\\n\\t\\theight: 16px;\\n\\t}\\n\\n\\t.overflow-item-selected :global(svg) {\\n\\t\\tcolor: var(--color-accent);\\n\\t}</style>\\n"],"names":[],"mappings":"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"}'
};
const TABS = {};
const Tabs = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let has_tabs;
  let $selected_tab, $$unsubscribe_selected_tab;
  let $selected_tab_index, $$unsubscribe_selected_tab_index;
  let { visible = true } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { selected } = $$props;
  let tabs = [];
  let overflow_menu_open = false;
  let overflow_menu;
  let tab_nav_el;
  let overflow_nav;
  const selected_tab = writable(false);
  $$unsubscribe_selected_tab = subscribe(selected_tab, (value) => $selected_tab = value);
  const selected_tab_index = writable(0);
  $$unsubscribe_selected_tab_index = subscribe(selected_tab_index, (value) => $selected_tab_index = value);
  const dispatch = createEventDispatcher();
  setContext(TABS, {
    register_tab: (tab) => {
      let index = tabs.findIndex((t) => t.id === tab.id);
      if (index !== -1) {
        tabs[index] = { ...tabs[index], ...tab };
      } else {
        tabs = [...tabs, tab];
        index = tabs.length - 1;
      }
      if ($selected_tab === false && tab.visible && tab.interactive) {
        set_store_value(selected_tab, $selected_tab = tab.id, $selected_tab);
      }
      return index;
    },
    unregister_tab: (tab) => {
      const index = tabs.findIndex((t) => t.id === tab.id);
      if (index !== -1) {
        tabs = tabs.filter((t) => t.id !== tab.id);
        if ($selected_tab === tab.id) {
          set_store_value(selected_tab, $selected_tab = tabs[0]?.id || false, $selected_tab);
        }
      }
    },
    selected_tab,
    selected_tab_index
  });
  function change_tab(id) {
    const tab_to_activate = tabs.find((t) => t.id === id);
    if (tab_to_activate && tab_to_activate.interactive && tab_to_activate.visible) {
      selected = id;
      set_store_value(selected_tab, $selected_tab = id, $selected_tab);
      set_store_value(selected_tab_index, $selected_tab_index = tabs.findIndex((t) => t.id === id), $selected_tab_index);
      dispatch("change");
      overflow_menu_open = false;
    } else {
      console.warn("Attempted to select a non-interactive or hidden tab.");
    }
  }
  onDestroy(() => {
    if (typeof window === "undefined")
      return;
    window.removeEventListener("resize", handle_menu_overflow);
    window.removeEventListener("click", handle_outside_click);
  });
  function handle_outside_click(event) {
    if (overflow_menu_open && overflow_menu && !overflow_menu.contains(event.target)) {
      overflow_menu_open = false;
    }
  }
  function handle_menu_overflow() {
    {
      console.error("Menu elements not found");
      return;
    }
  }
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.selected === void 0 && $$bindings.selected && selected !== void 0)
    $$bindings.selected(selected);
  $$result.css.add(css);
  has_tabs = tabs.length > 0;
  {
    selected !== null && change_tab(selected);
  }
  $$unsubscribe_selected_tab();
  $$unsubscribe_selected_tab_index();
  return `${has_tabs ? `<div class="${[
    "tabs " + escape(elem_classes.join(" "), true) + " svelte-r9stgs",
    !visible ? "hide" : ""
  ].join(" ").trim()}"${add_attribute("id", elem_id, 0)}><div class="tab-wrapper svelte-r9stgs"><div class="tab-container svelte-r9stgs" role="tablist"${add_attribute("this", tab_nav_el, 0)}>${each(tabs, (t, i) => {
    return `<button role="tab"${add_attribute("aria-selected", t.id === $selected_tab, 0)}${add_attribute("aria-controls", t.elem_id, 0)} ${!t.interactive ? "disabled" : ""}${add_attribute("aria-disabled", !t.interactive, 0)}${add_attribute("id", t.elem_id ? t.elem_id + "-button" : null, 0)}${add_attribute("data-tab-id", t.id, 0)} class="${["svelte-r9stgs", t.id === $selected_tab ? "selected" : ""].join(" ").trim()}">${escape(t.name)} </button>`;
  })}</div> <span class="${["overflow-menu svelte-r9stgs", "hide"].join(" ").trim()}"${add_attribute("this", overflow_menu, 0)}><button class="${[
    "svelte-r9stgs",
    ""
  ].join(" ").trim()}">${validate_component(OverflowIcon, "OverflowIcon").$$render($$result, {}, {}, {})}</button> <div class="${["overflow-dropdown svelte-r9stgs", !overflow_menu_open ? "hide" : ""].join(" ").trim()}"${add_attribute("this", overflow_nav, 0)}></div></span></div></div>` : ``} ${slots.default ? slots.default({}) : ``}`;
});
const Tabs$1 = Tabs;
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  const dispatch = createEventDispatcher();
  let { visible = true } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { selected } = $$props;
  let { gradio } = $$props;
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.selected === void 0 && $$bindings.selected && selected !== void 0)
    $$bindings.selected(selected);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    {
      dispatch("prop_change", { selected });
    }
    $$rendered = `${validate_component(Tabs$1, "Tabs").$$render(
      $$result,
      { visible, elem_id, elem_classes, selected },
      {
        selected: ($$value) => {
          selected = $$value;
          $$settled = false;
        }
      },
      {
        default: () => {
          return `${slots.default ? slots.default({}) : ``}`;
        }
      }
    )}`;
  } while (!$$settled);
  return $$rendered;
});

export { Tabs$1 as BaseTabs, TABS, Index as default };
//# sourceMappingURL=Index40-CIsHF6Ll.js.map
