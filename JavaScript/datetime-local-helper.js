/* From Stackoverflow assign javascript date to html5 datetime-local input */

export const getDateStringLocal = date => {
  const newDate = date ? new Date(date) : new Date()
  return new Date(newDate.getTime() - new Date.getTimezoneOffset() * 60000)
    .toISOString()
    .slice(0, -1)
}
