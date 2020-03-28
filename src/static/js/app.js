var data = []
var token = ""

jQuery(document).ready(function () {
    $('#btn-process').on('click', function () {
        review = $('#txt_review').val()
        $.ajax({
            url: '/predict',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "sentence": review
            }),
            beforeSend: function () {
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            $('#positive-score').val(jsondata['response']['positive'])
            $('#negative-score').val(jsondata['response']['negative'])
            $('#final-score').val(jsondata['response']['result'])
        }).fail(function (jsondata, textStatus, jqXHR) {
            alert(jsondata['responseJSON'])
        });
    })

    $('#txt_review').keypress(function (e) {
        if (e.which === 13) {
            $('#btn-process').click()
            e.preventDefault()
        }
    });
})