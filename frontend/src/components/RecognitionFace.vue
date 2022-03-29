<template>
  <v-container fluid ill-height class="pl-10">
    <v-row>
      <v-col cols="3">
        <v-select
          v-model="selectedClass"
          label="Class"
          :items="classes"
          outlined
          :menu-props="$menuProps"
        ></v-select>
      </v-col>
      <v-col cols="3">
        <v-select
          v-model="selectedImage"
          label="Image"
          :items="images"
          outlined
          :menu-props="$menuProps"
        ></v-select>
      </v-col>
    </v-row>
    <v-row>
      <v-col class="d-flex" cols="3">
        <v-select
          v-model="selectedMethod"
          :items="methods"
          label="Methods"
          item-text="name"
          item-value="id"
          :menu-props="$menuProps"
          width="250px"
          outlined
          hide-details
        ></v-select>
      </v-col>
      <v-col
        v-for="(item, index) in methods[selectedMethod].params"
        :key="index"
        cols="3"
      >
        <v-text-field outlined :label="item.name" hide-details></v-text-field>
      </v-col>
      <v-col cols="2" class="d-flex align-center">
        <v-btn class="elevation-1" color="success">Get result</v-btn>
      </v-col>
    </v-row>
  </v-container>
</template>
<script>
export default {
  data() {
    return {
      selectedMethod: 0,
      selectedClass: 1,
      selectedImage: 1,
      methods: [
        {
          id: 0,
          name: "Histogram",
          params: [
            {
              id: 1,
              name: "BIN",
            },
          ],
        },
        {
          id: 1,
          name: "DFT",
          params: [
            {
              id: 1,
              name: "P",
            },
            {
              id: 1,
              name: "Q",
            },
          ],
        },
        {
          id: 2,
          name: "DCT",
          params: [
            {
              id: 1,
              name: "P/Q",
            },
          ],
        },
        {
          id: 3,
          name: "Scale",
          params: [
            {
              id: 1,
              name: "scale",
            },
          ],
        },
        {
          id: 4,
          name: "Gradient",
          params: [
            {
              id: 1,
              name: "W",
            },
          ],
        },
      ],
    };
  },
  computed: {
    classes() {
      const data = [];
      [...Array(40)].forEach((e, i) => data.push(i + 1));
      return data;
    },
    images() {
      const data = [];
      [...Array(10)].forEach((e, i) => data.push(i + 1));
      return data;
    },
  },
};
</script>