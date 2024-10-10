import { SvelteComponent, init, safe_not_equal, svg_element, claim_svg_element, children, detach, attr, insert_hydration, append_hydration, noop, create_slot, space, claim_space, transition_in, group_outros, transition_out, check_outros, update_slot_base, get_all_dirty_from_scope, get_slot_changes, component_subscribe, createEventDispatcher, setContext, set_store_value, onMount, onDestroy, ensure_array_like, element, create_component, claim_element, claim_component, toggle_class, mount_component, listen, stop_propagation, update_keyed_each, destroy_block, destroy_component, text, claim_text, set_data, binding_callbacks } from "../../../svelte/svelte.js";
import { writable } from "../../../svelte/svelte-submodules.js";
function create_fragment$1(ctx) {
  let svg;
  let circle0;
  let circle1;
  let circle2;
  return {
    c() {
      svg = svg_element("svg");
      circle0 = svg_element("circle");
      circle1 = svg_element("circle");
      circle2 = svg_element("circle");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", {
        width: true,
        height: true,
        viewBox: true,
        fill: true,
        xmlns: true
      });
      var svg_nodes = children(svg);
      circle0 = claim_svg_element(svg_nodes, "circle", { cx: true, cy: true, r: true, fill: true });
      children(circle0).forEach(detach);
      circle1 = claim_svg_element(svg_nodes, "circle", { cx: true, cy: true, r: true, fill: true });
      children(circle1).forEach(detach);
      circle2 = claim_svg_element(svg_nodes, "circle", { cx: true, cy: true, r: true, fill: true });
      children(circle2).forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(circle0, "cx", "2.5");
      attr(circle0, "cy", "8");
      attr(circle0, "r", "1.5");
      attr(circle0, "fill", "currentColor");
      attr(circle1, "cx", "8");
      attr(circle1, "cy", "8");
      attr(circle1, "r", "1.5");
      attr(circle1, "fill", "currentColor");
      attr(circle2, "cx", "13.5");
      attr(circle2, "cy", "8");
      attr(circle2, "r", "1.5");
      attr(circle2, "fill", "currentColor");
      attr(svg, "width", "16");
      attr(svg, "height", "16");
      attr(svg, "viewBox", "0 0 16 16");
      attr(svg, "fill", "none");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, circle0);
      append_hydration(svg, circle1);
      append_hydration(svg, circle2);
    },
    p: noop,
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
class OverflowIcon extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, null, create_fragment$1, safe_not_equal, {});
  }
}
function get_each_context(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[27] = list[i];
  child_ctx[29] = i;
  return child_ctx;
}
function create_if_block(ctx) {
  let div3;
  let div2;
  let div0;
  let each_blocks = [];
  let each_1_lookup = /* @__PURE__ */ new Map();
  let t0;
  let span;
  let button;
  let overflowicon;
  let t1;
  let div1;
  let div3_class_value;
  let current;
  let mounted;
  let dispose;
  let each_value = ensure_array_like(
    /*tabs*/
    ctx[3]
  );
  const get_key = (ctx2) => (
    /*t*/
    ctx2[27].id
  );
  for (let i = 0; i < each_value.length; i += 1) {
    let child_ctx = get_each_context(ctx, each_value, i);
    let key = get_key(child_ctx);
    each_1_lookup.set(key, each_blocks[i] = create_each_block(key, child_ctx));
  }
  overflowicon = new OverflowIcon({});
  return {
    c() {
      div3 = element("div");
      div2 = element("div");
      div0 = element("div");
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      t0 = space();
      span = element("span");
      button = element("button");
      create_component(overflowicon.$$.fragment);
      t1 = space();
      div1 = element("div");
      this.h();
    },
    l(nodes) {
      div3 = claim_element(nodes, "DIV", { class: true, id: true });
      var div3_nodes = children(div3);
      div2 = claim_element(div3_nodes, "DIV", { class: true });
      var div2_nodes = children(div2);
      div0 = claim_element(div2_nodes, "DIV", { class: true, role: true });
      var div0_nodes = children(div0);
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(div0_nodes);
      }
      div0_nodes.forEach(detach);
      t0 = claim_space(div2_nodes);
      span = claim_element(div2_nodes, "SPAN", { class: true });
      var span_nodes = children(span);
      button = claim_element(span_nodes, "BUTTON", { class: true });
      var button_nodes = children(button);
      claim_component(overflowicon.$$.fragment, button_nodes);
      button_nodes.forEach(detach);
      t1 = claim_space(span_nodes);
      div1 = claim_element(span_nodes, "DIV", { class: true });
      children(div1).forEach(detach);
      span_nodes.forEach(detach);
      div2_nodes.forEach(detach);
      div3_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div0, "class", "tab-container svelte-r9stgs");
      attr(div0, "role", "tablist");
      attr(button, "class", "svelte-r9stgs");
      toggle_class(
        button,
        "overflow-item-selected",
        /*overflow_has_selected_tab*/
        ctx[9]
      );
      attr(div1, "class", "overflow-dropdown svelte-r9stgs");
      toggle_class(div1, "hide", !/*overflow_menu_open*/
      ctx[4]);
      attr(span, "class", "overflow-menu svelte-r9stgs");
      toggle_class(span, "hide", !/*is_overflowing*/
      ctx[8]);
      attr(div2, "class", "tab-wrapper svelte-r9stgs");
      attr(div3, "class", div3_class_value = "tabs " + /*elem_classes*/
      ctx[2].join(" ") + " svelte-r9stgs");
      attr(
        div3,
        "id",
        /*elem_id*/
        ctx[1]
      );
      toggle_class(div3, "hide", !/*visible*/
      ctx[0]);
    },
    m(target, anchor) {
      insert_hydration(target, div3, anchor);
      append_hydration(div3, div2);
      append_hydration(div2, div0);
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(div0, null);
        }
      }
      ctx[20](div0);
      append_hydration(div2, t0);
      append_hydration(div2, span);
      append_hydration(span, button);
      mount_component(overflowicon, button, null);
      append_hydration(span, t1);
      append_hydration(span, div1);
      ctx[22](div1);
      ctx[23](span);
      current = true;
      if (!mounted) {
        dispose = listen(button, "click", stop_propagation(
          /*click_handler_1*/
          ctx[21]
        ));
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (dirty & /*tabs, $selected_tab, change_tab, dispatch*/
      51208) {
        each_value = ensure_array_like(
          /*tabs*/
          ctx2[3]
        );
        each_blocks = update_keyed_each(each_blocks, dirty, get_key, 1, ctx2, each_value, each_1_lookup, div0, destroy_block, create_each_block, null, get_each_context);
      }
      if (!current || dirty & /*overflow_has_selected_tab*/
      512) {
        toggle_class(
          button,
          "overflow-item-selected",
          /*overflow_has_selected_tab*/
          ctx2[9]
        );
      }
      if (!current || dirty & /*overflow_menu_open*/
      16) {
        toggle_class(div1, "hide", !/*overflow_menu_open*/
        ctx2[4]);
      }
      if (!current || dirty & /*is_overflowing*/
      256) {
        toggle_class(span, "hide", !/*is_overflowing*/
        ctx2[8]);
      }
      if (!current || dirty & /*elem_classes*/
      4 && div3_class_value !== (div3_class_value = "tabs " + /*elem_classes*/
      ctx2[2].join(" ") + " svelte-r9stgs")) {
        attr(div3, "class", div3_class_value);
      }
      if (!current || dirty & /*elem_id*/
      2) {
        attr(
          div3,
          "id",
          /*elem_id*/
          ctx2[1]
        );
      }
      if (!current || dirty & /*elem_classes, visible*/
      5) {
        toggle_class(div3, "hide", !/*visible*/
        ctx2[0]);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(overflowicon.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(overflowicon.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div3);
      }
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].d();
      }
      ctx[20](null);
      destroy_component(overflowicon);
      ctx[22](null);
      ctx[23](null);
      mounted = false;
      dispose();
    }
  };
}
function create_each_block(key_1, ctx) {
  let button;
  let t0_value = (
    /*t*/
    ctx[27].name + ""
  );
  let t0;
  let t1;
  let button_aria_selected_value;
  let button_aria_controls_value;
  let button_disabled_value;
  let button_aria_disabled_value;
  let button_id_value;
  let button_data_tab_id_value;
  let mounted;
  let dispose;
  function click_handler() {
    return (
      /*click_handler*/
      ctx[19](
        /*t*/
        ctx[27],
        /*i*/
        ctx[29]
      )
    );
  }
  return {
    key: key_1,
    first: null,
    c() {
      button = element("button");
      t0 = text(t0_value);
      t1 = space();
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", {
        role: true,
        "aria-selected": true,
        "aria-controls": true,
        "aria-disabled": true,
        id: true,
        "data-tab-id": true,
        class: true
      });
      var button_nodes = children(button);
      t0 = claim_text(button_nodes, t0_value);
      t1 = claim_space(button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "role", "tab");
      attr(button, "aria-selected", button_aria_selected_value = /*t*/
      ctx[27].id === /*$selected_tab*/
      ctx[11]);
      attr(button, "aria-controls", button_aria_controls_value = /*t*/
      ctx[27].elem_id);
      button.disabled = button_disabled_value = !/*t*/
      ctx[27].interactive;
      attr(button, "aria-disabled", button_aria_disabled_value = !/*t*/
      ctx[27].interactive);
      attr(button, "id", button_id_value = /*t*/
      ctx[27].elem_id ? (
        /*t*/
        ctx[27].elem_id + "-button"
      ) : null);
      attr(button, "data-tab-id", button_data_tab_id_value = /*t*/
      ctx[27].id);
      attr(button, "class", "svelte-r9stgs");
      toggle_class(
        button,
        "selected",
        /*t*/
        ctx[27].id === /*$selected_tab*/
        ctx[11]
      );
      this.first = button;
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      append_hydration(button, t0);
      append_hydration(button, t1);
      if (!mounted) {
        dispose = listen(button, "click", click_handler);
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if (dirty & /*tabs*/
      8 && t0_value !== (t0_value = /*t*/
      ctx[27].name + ""))
        set_data(t0, t0_value);
      if (dirty & /*tabs, $selected_tab*/
      2056 && button_aria_selected_value !== (button_aria_selected_value = /*t*/
      ctx[27].id === /*$selected_tab*/
      ctx[11])) {
        attr(button, "aria-selected", button_aria_selected_value);
      }
      if (dirty & /*tabs*/
      8 && button_aria_controls_value !== (button_aria_controls_value = /*t*/
      ctx[27].elem_id)) {
        attr(button, "aria-controls", button_aria_controls_value);
      }
      if (dirty & /*tabs*/
      8 && button_disabled_value !== (button_disabled_value = !/*t*/
      ctx[27].interactive)) {
        button.disabled = button_disabled_value;
      }
      if (dirty & /*tabs*/
      8 && button_aria_disabled_value !== (button_aria_disabled_value = !/*t*/
      ctx[27].interactive)) {
        attr(button, "aria-disabled", button_aria_disabled_value);
      }
      if (dirty & /*tabs*/
      8 && button_id_value !== (button_id_value = /*t*/
      ctx[27].elem_id ? (
        /*t*/
        ctx[27].elem_id + "-button"
      ) : null)) {
        attr(button, "id", button_id_value);
      }
      if (dirty & /*tabs*/
      8 && button_data_tab_id_value !== (button_data_tab_id_value = /*t*/
      ctx[27].id)) {
        attr(button, "data-tab-id", button_data_tab_id_value);
      }
      if (dirty & /*tabs, $selected_tab*/
      2056) {
        toggle_class(
          button,
          "selected",
          /*t*/
          ctx[27].id === /*$selected_tab*/
          ctx[11]
        );
      }
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      mounted = false;
      dispose();
    }
  };
}
function create_fragment(ctx) {
  let t_1;
  let current;
  let if_block = (
    /*has_tabs*/
    ctx[10] && create_if_block(ctx)
  );
  const default_slot_template = (
    /*#slots*/
    ctx[18].default
  );
  const default_slot = create_slot(
    default_slot_template,
    ctx,
    /*$$scope*/
    ctx[17],
    null
  );
  return {
    c() {
      if (if_block)
        if_block.c();
      t_1 = space();
      if (default_slot)
        default_slot.c();
    },
    l(nodes) {
      if (if_block)
        if_block.l(nodes);
      t_1 = claim_space(nodes);
      if (default_slot)
        default_slot.l(nodes);
    },
    m(target, anchor) {
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, t_1, anchor);
      if (default_slot) {
        default_slot.m(target, anchor);
      }
      current = true;
    },
    p(ctx2, [dirty]) {
      if (
        /*has_tabs*/
        ctx2[10]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*has_tabs*/
          1024) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(t_1.parentNode, t_1);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      if (default_slot) {
        if (default_slot.p && (!current || dirty & /*$$scope*/
        131072)) {
          update_slot_base(
            default_slot,
            default_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[17],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[17]
            ) : get_slot_changes(
              default_slot_template,
              /*$$scope*/
              ctx2[17],
              dirty,
              null
            ),
            null
          );
        }
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      transition_in(default_slot, local);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      transition_out(default_slot, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t_1);
      }
      if (if_block)
        if_block.d(detaching);
      if (default_slot)
        default_slot.d(detaching);
    }
  };
}
const TABS = {};
function instance($$self, $$props, $$invalidate) {
  let has_tabs;
  let $selected_tab;
  let $selected_tab_index;
  let { $$slots: slots = {}, $$scope } = $$props;
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
  component_subscribe($$self, selected_tab, (value) => $$invalidate(11, $selected_tab = value));
  const selected_tab_index = writable(0);
  component_subscribe($$self, selected_tab_index, (value) => $$invalidate(24, $selected_tab_index = value));
  const dispatch = createEventDispatcher();
  let is_overflowing = false;
  let overflow_has_selected_tab = false;
  setContext(TABS, {
    register_tab: (tab) => {
      let index = tabs.findIndex((t) => t.id === tab.id);
      if (index !== -1) {
        $$invalidate(3, tabs[index] = { ...tabs[index], ...tab }, tabs);
      } else {
        $$invalidate(3, tabs = [...tabs, tab]);
        index = tabs.length - 1;
      }
      if ($selected_tab === false && tab.visible && tab.interactive) {
        set_store_value(selected_tab, $selected_tab = tab.id, $selected_tab);
      }
      return index;
    },
    unregister_tab: (tab) => {
      var _a;
      const index = tabs.findIndex((t) => t.id === tab.id);
      if (index !== -1) {
        $$invalidate(3, tabs = tabs.filter((t) => t.id !== tab.id));
        if ($selected_tab === tab.id) {
          set_store_value(selected_tab, $selected_tab = ((_a = tabs[0]) == null ? void 0 : _a.id) || false, $selected_tab);
        }
      }
    },
    selected_tab,
    selected_tab_index
  });
  function change_tab(id) {
    const tab_to_activate = tabs.find((t) => t.id === id);
    if (tab_to_activate && tab_to_activate.interactive && tab_to_activate.visible) {
      $$invalidate(16, selected = id);
      set_store_value(selected_tab, $selected_tab = id, $selected_tab);
      set_store_value(selected_tab_index, $selected_tab_index = tabs.findIndex((t) => t.id === id), $selected_tab_index);
      dispatch("change");
      $$invalidate(4, overflow_menu_open = false);
    } else {
      console.warn("Attempted to select a non-interactive or hidden tab.");
    }
  }
  onMount(() => {
    handle_menu_overflow();
    window.addEventListener("resize", handle_menu_overflow);
    window.addEventListener("click", handle_outside_click);
  });
  onDestroy(() => {
    if (typeof window === "undefined")
      return;
    window.removeEventListener("resize", handle_menu_overflow);
    window.removeEventListener("click", handle_outside_click);
  });
  function handle_outside_click(event) {
    if (overflow_menu_open && overflow_menu && !overflow_menu.contains(event.target)) {
      $$invalidate(4, overflow_menu_open = false);
    }
  }
  function handle_menu_overflow() {
    if (!tab_nav_el) {
      console.error("Menu elements not found");
      return;
    }
    let all_items = [];
    [tab_nav_el, overflow_nav].forEach((menu) => {
      Array.from(menu.querySelectorAll("button")).forEach((item) => all_items.push(item));
    });
    all_items.forEach((item) => tab_nav_el.appendChild(item));
    const nav_items = [];
    const overflow_items = [];
    Array.from(tab_nav_el.querySelectorAll("button")).forEach((item) => {
      const tab_rect = item.getBoundingClientRect();
      const tab_menu_rect = tab_nav_el.getBoundingClientRect();
      $$invalidate(8, is_overflowing = tab_rect.right > tab_menu_rect.right || tab_rect.left < tab_menu_rect.left);
      if (is_overflowing) {
        overflow_items.push(item);
      } else {
        nav_items.push(item);
      }
    });
    nav_items.forEach((item) => tab_nav_el.appendChild(item));
    overflow_items.forEach((item) => overflow_nav.appendChild(item));
    $$invalidate(9, overflow_has_selected_tab = tabs.some((t) => t.id === $selected_tab && overflow_nav.contains(document.querySelector(`[data-tab-id="${t.id}"]`))));
  }
  const click_handler = (t, i) => {
    if (t.id !== $selected_tab) {
      change_tab(t.id);
      dispatch("select", { value: t.name, index: i });
    }
  };
  function div0_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      tab_nav_el = $$value;
      $$invalidate(6, tab_nav_el);
    });
  }
  const click_handler_1 = () => $$invalidate(4, overflow_menu_open = !overflow_menu_open);
  function div1_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      overflow_nav = $$value;
      $$invalidate(7, overflow_nav);
    });
  }
  function span_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      overflow_menu = $$value;
      $$invalidate(5, overflow_menu);
    });
  }
  $$self.$$set = ($$props2) => {
    if ("visible" in $$props2)
      $$invalidate(0, visible = $$props2.visible);
    if ("elem_id" in $$props2)
      $$invalidate(1, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(2, elem_classes = $$props2.elem_classes);
    if ("selected" in $$props2)
      $$invalidate(16, selected = $$props2.selected);
    if ("$$scope" in $$props2)
      $$invalidate(17, $$scope = $$props2.$$scope);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*tabs*/
    8) {
      $$invalidate(10, has_tabs = tabs.length > 0);
    }
    if ($$self.$$.dirty & /*tabs, selected*/
    65544) {
      selected !== null && change_tab(selected);
    }
  };
  return [
    visible,
    elem_id,
    elem_classes,
    tabs,
    overflow_menu_open,
    overflow_menu,
    tab_nav_el,
    overflow_nav,
    is_overflowing,
    overflow_has_selected_tab,
    has_tabs,
    $selected_tab,
    selected_tab,
    selected_tab_index,
    dispatch,
    change_tab,
    selected,
    $$scope,
    slots,
    click_handler,
    div0_binding,
    click_handler_1,
    div1_binding,
    span_binding
  ];
}
class Tabs extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      visible: 0,
      elem_id: 1,
      elem_classes: 2,
      selected: 16
    });
  }
}
const Tabs$1 = Tabs;
export {
  Tabs$1 as T,
  TABS as a
};
