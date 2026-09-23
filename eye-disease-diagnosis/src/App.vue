<template>
  <div id="app">
    <!-- 通过路由元信息控制组件显示 -->
    <HeaderNav v-if="!$route.meta.hideLayout" />
    <router-view style="flex-grow: 1;" />
    <Footer v-if="!$route.meta.hideLayout" />
  </div>
</template>

<script>
import Footer from './components/Footer.vue';
import HeaderNav from './components/HeaderNav.vue';
export default {
  name: 'App',
  components: {
    HeaderNav,
    Footer,
  },

}
</script>

<style>

#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  justify-content: space-between;
  margin: 0;
  padding: 0;
}

/* 添加响应式样式
@media (max-width: 768px) {
  #app {
    padding: 0 1rem;
  }
} */

</style>