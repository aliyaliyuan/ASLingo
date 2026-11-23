// Everything the store sells. `section` is the id of the zone it is found in
// (see ZONES in pages/Game.jsx). Edit this list to change the vocabulary.
export const ITEMS = [
  { id: 'apple', word: 'Apple', icon: '🍎', section: 'fruit' },
  { id: 'orange', word: 'Orange', icon: '🍊', section: 'fruit' },
  { id: 'strawberry', word: 'Strawberry', icon: '🍓', section: 'fruit' },
  { id: 'carrot', word: 'Carrot', icon: '🥕', section: 'vegetable' },
  { id: 'broccoli', word: 'Broccoli', icon: '🥦', section: 'vegetable' },
  { id: 'lettuce', word: 'Lettuce', icon: '🥬', section: 'vegetable' },
  { id: 'cake', word: 'Cake', icon: '🍰', section: 'dessert' },
  { id: 'cookie', word: 'Cookie', icon: '🍪', section: 'dessert' },
]

// A new shopping list: one random item from each section.
export function makeShoppingList() {
  const sections = [...new Set(ITEMS.map((item) => item.section))]
  return sections.map((section) => {
    const choices = ITEMS.filter((item) => item.section === section)
    return choices[Math.floor(Math.random() * choices.length)]
  })
}
